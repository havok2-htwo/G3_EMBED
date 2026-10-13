"""CUDA graphs for short EmbeddingGemma 2 text passes.

Short texts are launch-bound: a compiled forward of one 20-token query spends ~1.6 ms on the GPU but ~28 ms
on the CPU launching ~400 Triton kernels and ~220 matmuls (RTX 4090, Windows/WDDM). Replaying a captured
CUDA graph removes that CPU cost.

The text forward of the inner model (``EmbeddingGemma2Model.forward`` with ``input_ids`` and
``attention_mask`` only) is replaced by an instance attribute. Inputs are right-padded to a (rows, tokens)
bucket; padded keys are masked out and only the real rows/tokens are returned, so pooling sees the same
shape as eager. The attention masks are passed pre-built (``{"full_attention", "sliding_attention"}``), which
the text model accepts and which keeps the data-dependent "is the mask all ones" check out of the graph.

- All graphs share one memory pool, one side stream, one static input buffer and one static output buffer
  (only one pass runs at a time: the caller holds the model's encode_lock). Largest bucket first, so the
  smaller ones reuse its pool.
- Capture happens once, at model load, on the loading thread (``capture_error_mode="thread_local"`` so other
  models running on other threads are not affected). Missing buckets, media, other kwargs, longer
  sequences or a failed capture run the original forward.
- A VRAM budget stops capturing further buckets when exceeded.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Sequence

# Token buckets up to the sliding window (512): within it the sliding mask equals the full mask.
SEQ_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
ROW_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Most padded tokens a graph may cover; larger passes are compute-bound and run eager.
MAX_GRAPH_TOKENS = 4096
DEFAULT_MEMORY_BUDGET_MB = 1024

_TEXT_KWARGS = frozenset({"input_ids", "attention_mask", "return_dict"})


def bucket_for(value: int, buckets: Sequence[int]) -> int | None:
    for bucket in buckets:
        if value <= bucket:
            return bucket
    return None


def graph_buckets(max_tokens: int = MAX_GRAPH_TOKENS) -> list[tuple[int, int]]:
    """(rows, tokens) buckets to capture, largest first."""
    pairs = [(rows, tokens) for tokens in SEQ_BUCKETS for rows in ROW_BUCKETS if rows * tokens <= max_tokens]
    return sorted(pairs, key=lambda pair: (pair[0] * pair[1], pair[1]), reverse=True)


def supported(inner_model: Any) -> tuple[bool, str]:
    """Whether the inner transformers model is one this module knows how to capture."""
    import torch

    if not torch.cuda.is_available():
        return False, "CUDA not available"
    config = getattr(inner_model, "config", None)
    if getattr(config, "model_type", "") != "embedding_gemma2":
        return False, f"model type '{getattr(config, 'model_type', '')}' not supported"
    try:
        parameter = next(inner_model.parameters())
    except StopIteration:
        return False, "no parameters"
    if parameter.device.type != "cuda":
        return False, f"model on '{parameter.device}'"
    if parameter.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {parameter.dtype} not supported"
    if getattr(inner_model, "hf_quantizer", None) is not None:
        return False, "quantized model"
    if getattr(getattr(config, "text_config", None), "sliding_window", None) is None:
        return False, "no sliding_window in the text config"
    return True, ""


class TextGraphRunner:
    """Captured text forwards of one EmbeddingGemma 2 model, installed as its ``forward``."""

    def __init__(self, inner_model: Any, memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB, max_tokens: int = MAX_GRAPH_TOKENS):
        import torch

        self.inner_model = inner_model
        self.original_forward: Callable[..., Any] = inner_model.forward
        parameter = next(inner_model.parameters())
        self.device = parameter.device
        text_config = inner_model.config.text_config
        self.sliding_window = int(text_config.sliding_window)
        self.pad_token_id = int(getattr(text_config, "pad_token_id", 0) or 0)
        self.memory_budget_bytes = int(memory_budget_mb) * 1024 * 1024
        self.max_tokens = int(max_tokens)
        self.graphs: dict[tuple[int, int], Any] = {}
        self.output_type: Any = None
        self.memory_bytes = 0
        self.capture_ms = 0.0
        self.replays = 0
        self.fallbacks = 0
        self.disabled_reason: str | None = None
        self._lock = threading.Lock()
        self._stream = torch.cuda.Stream(device=self.device)
        self._pool = torch.cuda.graph_pool_handle()
        self._ids = torch.full((self.max_tokens,), self.pad_token_id, dtype=torch.long, device=self.device)
        self._mask = torch.zeros((self.max_tokens,), dtype=torch.long, device=self.device)
        self._out: Any = None  # allocated on first capture (hidden size from the model output)

    # --- capture -------------------------------------------------------------------------------------------

    def _views(self, rows: int, tokens: int):
        count = rows * tokens
        return self._ids[:count].view(rows, tokens), self._mask[:count].view(rows, tokens)

    def _masks(self, mask2d: Any) -> dict[str, Any]:
        import torch

        rows, tokens = mask2d.shape
        keys = mask2d.bool()[:, None, None, :].expand(rows, 1, tokens, tokens)
        positions = torch.arange(tokens, device=mask2d.device)
        window = (positions[:, None] - positions[None, :]).abs() <= self.sliding_window
        return {"full_attention": keys, "sliding_attention": keys & window[None, None, :, :]}

    def _run(self, rows: int, tokens: int) -> Any:
        ids, mask = self._views(rows, tokens)
        output = self.original_forward(input_ids=ids, attention_mask=self._masks(mask), return_dict=True)
        if self.output_type is None:
            self.output_type = type(output)
        hidden = output.last_hidden_state
        out = self._out[: rows * tokens * hidden.shape[-1]].view(rows, tokens, hidden.shape[-1])
        out.copy_(hidden)
        return out

    def capture(self, buckets: Sequence[tuple[int, int]] | None = None) -> None:
        """Capture every bucket (largest first) until the VRAM budget is reached. Call on the loading thread."""
        import torch

        started = time.perf_counter()
        # Process-local (mem_get_info also moves with the other services on the GPU).
        reserved_before = torch.cuda.memory_reserved(self.device) - self.memory_bytes
        dtype = next(self.inner_model.parameters()).dtype
        with torch.inference_mode():
            if self._out is None:
                # Probe the hidden size with one eager pass of the smallest bucket.
                ids, mask = self._views(1, SEQ_BUCKETS[0])
                mask.fill_(1)
                probe = self.original_forward(input_ids=ids, attention_mask=self._masks(mask), return_dict=True)
                self._out = torch.empty(self.max_tokens * probe.last_hidden_state.shape[-1], dtype=dtype, device=self.device)
                del probe
            for rows, tokens in buckets if buckets is not None else graph_buckets(self.max_tokens):
                if (rows, tokens) in self.graphs:
                    continue
                if self.memory_bytes > self.memory_budget_bytes:
                    print(f"[G3_EMBED] CUDA graphs: memory budget reached, {len(self.graphs)} buckets captured", flush=True)
                    break
                ids, mask = self._views(rows, tokens)
                ids.fill_(self.pad_token_id)
                mask.fill_(1)
                self._stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self._stream):
                    # Warm run outside the graph: torch.compile specializes/compiles here, not during capture.
                    self._run(rows, tokens)
                    self._run(rows, tokens)
                self._stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._pool, stream=self._stream, capture_error_mode="thread_local"):
                    self._run(rows, tokens)
                graph.replay()
                self._stream.synchronize()
                self.graphs[(rows, tokens)] = graph
                self.memory_bytes = max(0, torch.cuda.memory_reserved(self.device) - reserved_before)
        torch.cuda.synchronize(self.device)
        self.capture_ms = (time.perf_counter() - started) * 1000

    # --- dispatch ------------------------------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if (
            args
            or not self.graphs
            or self.disabled_reason
            or not set(kwargs) <= _TEXT_KWARGS
            or kwargs.get("return_dict") is False
            or not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or input_ids.dim() != 2
            or input_ids.shape != attention_mask.shape
            or input_ids.device != self.device
        ):
            return self._fallback(args, kwargs)
        rows, tokens = input_ids.shape
        token_bucket = bucket_for(tokens, SEQ_BUCKETS)
        row_bucket = bucket_for(rows, ROW_BUCKETS)
        graph = self.graphs.get((row_bucket, token_bucket)) if token_bucket and row_bucket else None
        if graph is None:
            return self._fallback(args, kwargs)
        with self._lock:
            ids, mask = self._views(row_bucket, token_bucket)
            ids.fill_(self.pad_token_id)
            mask.zero_()
            # Padded rows attend to one (pad) token so no attention row is fully masked.
            mask[:, 0] = 1
            ids[:rows, :tokens].copy_(input_ids)
            mask[:rows, :tokens].copy_(attention_mask)
            graph.replay()
            hidden_size = self._out.numel() // self.max_tokens
            out = self._out[: row_bucket * token_bucket * hidden_size].view(row_bucket, token_bucket, hidden_size)
            hidden = out[:rows, :tokens].clone()
            self.replays += 1
        return self.output_type(last_hidden_state=hidden)

    def _fallback(self, args: tuple, kwargs: dict) -> Any:
        self.fallbacks += 1
        return self.original_forward(*args, **kwargs)

    def install(self) -> None:
        self.inner_model.forward = self.forward

    def uninstall(self) -> None:
        if self.inner_model.__dict__.get("forward") == self.forward:
            del self.inner_model.forward

    def status(self) -> dict[str, Any]:
        return {
            "graphs": len(self.graphs),
            "memory_mb": round(self.memory_bytes / (1024 * 1024), 1),
            "capture_ms": round(self.capture_ms, 1),
            "replays": self.replays,
            "fallbacks": self.fallbacks,
        }
