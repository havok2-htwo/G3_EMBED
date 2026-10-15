from __future__ import annotations

import gc
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from .genesis_embed_server_globals import (
    DEFAULT_MODEL_ID,
    MAX_SEQ_LENGTH_LIMIT,
    current_settings,
    get_model_spec,
    resolve_model_cache_path,
    settings_lock,
)
from .genesis_embed_server_hardware import RuntimeChoice, get_system_snapshot, resolve_runtime_choice
from .genesis_embed_server_media import (
    DecodedMedia,
    MediaLimits,
    estimate_pass_mb,
    input_tokens,
    item_cost,
    media_summary,
    plan_chunks,
)

# Free memory (beyond the pass estimate) a media pass must leave for the other services on the GPU.
CUDA_MEDIA_MARGIN_MB = 256
# Most texts per forward pass when packing by tokens. Short texts are overhead-bound (EmbeddingGemma 2 on an
# RTX 4090: ~25 ms per pass whether it holds 4 or 64 short texts); beyond 128 the gain is small.
TOKEN_BATCH_MAX_ROWS = 128


@dataclass(frozen=True)
class RuntimeKey:
    model_id: str
    backend: str
    device: str
    precision: str
    compile_mode: str
    quantization: str = "none"
    cuda_graphs: bool = False


@dataclass
class EncodeResult:
    model_id: str
    dimension: int
    vectors: List[List[float]]
    backend: str
    device: str
    device_label: str
    load_duration_ms: float
    encode_duration_ms: float
    total_duration_ms: float
    texts_per_second: float | None
    input_count: int
    input_chars: int
    input_tokens: int | None = None
    # {"images": n, "audio_seconds": x, "video_frames": n} for requests with media.
    media: dict[str, Any] | None = None


class GpuOutOfMemory(RuntimeError):
    """CUDA ran out of memory during a forward pass (the cache has been released)."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"GPU out of memory while encoding with '{model_id}'")


class GpuMemoryUnavailable(RuntimeError):
    """Not enough free GPU memory for a media pass; it was not attempted."""

    def __init__(self, required_mb: float, available_mb: float):
        self.required_mb = round(required_mb, 1)
        self.available_mb = round(available_mb, 1)
        super().__init__(f"{self.available_mb} MB GPU memory available, {self.required_mb} MB needed")


_model_cache_lock = threading.RLock()
_model_cache: dict[RuntimeKey, dict[str, Any]] = {}
_last_runtime_error: str | None = None
_sdpa_gqa_disabled = False



class _DropMatMul8bitCastWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "MatMul8bitLt: inputs will be cast" not in str(record.msg)


# bitsandbytes INT8 logs on every matmul that bf16 inputs are cast to fp16.
logging.getLogger("bitsandbytes.autograd._functions").addFilter(_DropMatMul8bitCastWarning())


def _resolve_token() -> str | None:
    with settings_lock:
        token = str(current_settings.get("huggingface_token", "")).strip()
    return token or None


def _is_cuda_pytorch(choice: RuntimeChoice) -> bool:
    return choice.backend == "pytorch" and choice.device.startswith("cuda")


def _disable_sdpa_gqa() -> None:
    """Keep GQA models (Qwen3) on the memory-efficient SDPA kernel.

    For unpadded batches transformers calls SDPA with ``enable_gqa=True``. The
    memory-efficient kernel rejects that, so the dispatcher falls back to the
    math kernel, which materializes the full heads x seq x seq attention matrix
    (~3 GB per 4096-token sequence). Flash attention is missing from Windows
    builds. cuDNN attention accepts GQA but builds a new plan for every unseen
    sequence length (~6x slower on varied inputs). Expanding K/V via repeat_kv
    instead costs only a few MB per layer.
    """
    global _sdpa_gqa_disabled
    if _sdpa_gqa_disabled:
        return
    _sdpa_gqa_disabled = True
    try:
        from transformers.integrations import sdpa_attention

        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
    except Exception as exc:
        print(f"[G3_EMBED] SDPA GQA override not applied: {exc}", flush=True)


def _model_kwargs_for_precision(
    choice: RuntimeChoice,
    precision: str,
    quantization: str = "none",
    preferred_dtype: str | None = None,
) -> dict[str, Any]:
    if choice.backend != "pytorch":
        return {}
    try:
        import torch
    except Exception:
        return {}

    if choice.device == "cpu":
        # transformers 5 defaults to the checkpoint dtype (bf16 for Gemma/Qwen3), which is slow on most CPUs.
        return {"dtype": torch.bfloat16 if precision == "bfloat16" else torch.float32}

    selected = precision
    if selected == "auto":
        if not choice.device.startswith("cuda"):
            selected = "float32"
        elif preferred_dtype == "bfloat16" and torch.cuda.get_device_capability(torch.device(choice.device))[0] >= 8:
            selected = "bfloat16"
        else:
            selected = "float16"
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_by_name.get(selected)
    kwargs: dict[str, Any] = {"dtype": torch_dtype} if torch_dtype is not None else {}

    if quantization in {"int8", "nf4"}:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(f"Quantization '{quantization}' requires bitsandbytes support in transformers: {exc}") from exc
        if quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return kwargs


def _transformer_layer_stack(module: Any) -> Any:
    """Largest ModuleList of text transformer layers (skips vision/audio towers of multimodal models)."""
    import torch

    candidates = [
        child
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.ModuleList)
        and len(child) >= 2
        and not any(part in name for part in ("vision", "audio"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda layers: sum(parameter.numel() for parameter in layers.parameters()))


def _maybe_compile_sentence_transformer(model: Any, compile_mode: str, choice: RuntimeChoice, quantization: str = "none") -> bool:
    """Compile in place. "regional" compiles each transformer layer (fast startup, reused graph);
    "full" compiles the whole model (highest throughput, much longer warmup). Returns True if compiled."""
    if compile_mode == "off" or choice.backend != "pytorch" or choice.device == "cpu":
        return False
    if quantization != "none":
        print("[G3_EMBED] torch.compile skipped: not supported with bitsandbytes quantization", flush=True)
        return False
    try:
        import importlib.util

        import torch

        if not hasattr(torch, "compile"):
            return False
        # Inductor needs Triton on CUDA/XPU (triton-windows on Windows); compile would fail on first encode.
        if importlib.util.find_spec("triton") is None:
            print("[G3_EMBED] torch.compile skipped: Triton is not installed", flush=True)
            return False
        import torch._dynamo

        # Fall back to eager instead of failing requests if a graph cannot be compiled.
        torch._dynamo.config.suppress_errors = True
        # The default budget (8 per code object) is shared across all models loaded in this process.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 1024)
        first_module = model._first_module()
        # sentence-transformers 6 exposes auto_model read-only; the module itself lives in .model.
        attribute = "model" if isinstance(getattr(first_module, "model", None), torch.nn.Module) else "auto_model"
        inner_model = getattr(first_module, attribute, None)
        if inner_model is None:
            return False
        # Embeddings never need a KV cache; with it, each Qwen3 layer guards on its own layer index and
        # regional compile builds one graph per layer instead of one shared graph.
        config = getattr(inner_model, "config", None)
        for sub_config in (config, getattr(config, "text_config", None)):
            if sub_config is not None and hasattr(sub_config, "use_cache"):
                sub_config.use_cache = False
        language_model = getattr(inner_model, "language_model", None)
        has_towers = any(getattr(inner_model, name, None) is not None for name in ("vision_tower", "audio_tower"))
        if compile_mode == "regional":
            layers = _transformer_layer_stack(inner_model)
            if layers is None:
                print("[G3_EMBED] torch.compile skipped: no transformer layer stack found", flush=True)
                return False
            for layer in layers:
                layer.compile(dynamic=True)
        elif has_towers and isinstance(language_model, torch.nn.Module):
            # Only the language model: compiled vision/audio towers would recompile (and spike VRAM) per media shape.
            language_model.compile(dynamic=True)
        else:
            setattr(first_module, attribute, torch.compile(inner_model, dynamic=True))
        return True
    except Exception as exc:
        print(f"[G3_EMBED] torch.compile skipped: {exc}", flush=True)
        return False


def _warmup_compiled_model(model: Any, max_seq_length: int | None, encode_batch_size: int) -> None:
    """Trigger compilation at load time instead of on the first requests.

    Batch size 1 is specialized, and attention masks differ between short/long and padded/unpadded
    batches (sliding-window layers), so warm up each combination the configured batch size can produce.
    """
    started = time.perf_counter()
    if max_seq_length:
        model.max_seq_length = max_seq_length
    long_text = "warmup " * 2048
    batches: list[list[str]] = [["warmup"], [long_text]]
    if encode_batch_size >= 2:
        batches += [["warmup", long_text], ["warmup", "warmup"], [long_text, long_text]]
    try:
        for texts in batches:
            model.encode(_wrap_texts(texts), batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
    except Exception as exc:
        # Keep the model usable; anything not compiled here compiles lazily on first use.
        print(f"[G3_EMBED] torch.compile warmup failed: {exc}", flush=True)
        _release_cuda_cache()
        return
    print(f"[G3_EMBED] torch.compile warmup finished in {time.perf_counter() - started:.1f} s", flush=True)


def _skip_redundant_to_and_eval(model: Any) -> None:
    """Make sentence-transformers' per-call model.to(device) and model.eval() return early.

    encode() calls both on every call and each walks every module (~1500 with the vision/audio towers of
    EmbeddingGemma 2): ~11 ms per call on a Ryzen 5900X, a third of a short request. Models are never moved
    or trained after loading, so a same-device to() and eval() of an eval-mode model are no-ops.
    """
    import torch

    original_to = model.to
    original_eval = model.eval

    def to(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs and isinstance(args[0], (str, torch.device)):
            try:
                if torch.device(args[0]) == model.device:
                    return model
            except Exception:
                pass
        return original_to(*args, **kwargs)

    def eval_() -> Any:
        if not model.training:
            return model
        return original_eval()

    model.to = to
    model.eval = eval_


def _install_cuda_graphs(model: Any) -> Any:
    """Capture CUDA graphs for short text passes (EmbeddingGemma 2 on CUDA); returns the runner or None."""
    from . import genesis_embed_server_cuda_graphs as cuda_graphs

    first_module = model._first_module()
    inner_model = getattr(first_module, "model", None)
    ok, reason = cuda_graphs.supported(inner_model)
    if not ok:
        print(f"[G3_EMBED] CUDA graphs skipped: {reason}", flush=True)
        return None
    runner = cuda_graphs.TextGraphRunner(inner_model)
    try:
        runner.capture()
    except Exception as exc:
        print(f"[G3_EMBED] CUDA graphs disabled, capture failed: {exc}", flush=True)
        runner.graphs.clear()
        _release_cuda_cache()
        return None
    runner.install()
    status = runner.status()
    print(
        f"[G3_EMBED] CUDA graphs: {status['graphs']} text buckets captured in {status['capture_ms'] / 1000:.1f} s, "
        f"{status['memory_mb']:.0f} MB",
        flush=True,
    )
    return runner


def _load_model(runtime_key: RuntimeKey, settings: dict[str, Any], choice: RuntimeChoice) -> tuple[Any, float]:
    global _last_runtime_error

    cached = _model_cache.get(runtime_key)
    if cached is not None:
        return cached["model"], 0.0

    _evict_model_variants(runtime_key)
    if _is_cuda_pytorch(choice):
        _disable_sdpa_gqa()

    load_started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        cache_folder = resolve_model_cache_path(settings.get("model_cache_path", ""))
        kwargs: dict[str, Any] = {
            "cache_folder": cache_folder or None,
            "token": _resolve_token(),
        }
        if choice.backend == "openvino":
            kwargs["backend"] = "openvino"
            kwargs["device"] = choice.device
        else:
            kwargs["device"] = choice.device
            model_kwargs = _model_kwargs_for_precision(
                choice,
                runtime_key.precision,
                runtime_key.quantization,
                get_model_spec(runtime_key.model_id).get("preferred_dtype"),
            )
            if model_kwargs:
                kwargs["model_kwargs"] = model_kwargs

        model = SentenceTransformer(runtime_key.model_id, **kwargs)
        # Some tokenizers report a placeholder limit (EmbeddingGemma 2: 1e30); cap it at the spec or global limit.
        spec_limit = int(get_model_spec(runtime_key.model_id).get("max_seq_length") or MAX_SEQ_LENGTH_LIMIT)
        native_max_seq_length = min(int(getattr(model, "max_seq_length", 0) or 0) or spec_limit, spec_limit)
        compiled = _maybe_compile_sentence_transformer(model, runtime_key.compile_mode, choice, runtime_key.quantization)
        if _is_cuda_pytorch(choice):
            # Quantized loading stages full-precision shards on the GPU; release that cache.
            _release_cuda_cache()

        if choice.backend == "pytorch":
            _skip_redundant_to_and_eval(model)
        tokenizer_lock = threading.RLock()
        _serialize_preprocess(model, tokenizer_lock)

        if compiled:
            _warmup_compiled_model(
                model,
                _effective_max_seq_length(runtime_key.model_id, settings, native_max_seq_length),
                _effective_encode_batch_size(runtime_key.model_id, settings, 2),
            )
        elif settings.get("warmup_on_load"):
            model.encode(_wrap_texts(["warmup"]), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        graph_runner = _install_cuda_graphs(model) if runtime_key.cuda_graphs else None

        load_ms = (time.perf_counter() - load_started) * 1000
        _model_cache[runtime_key] = {
            "model": model,
            "loaded_at": time.time(),
            "load_duration_ms": load_ms,
            "choice": choice,
            "native_max_seq_length": native_max_seq_length,
            "encode_lock": threading.Lock(),
            "tokenizer_lock": tokenizer_lock,
            "cuda_graphs": graph_runner,
        }
        _last_runtime_error = None
        print(
            f"[G3_EMBED] Loaded '{runtime_key.model_id}' via {choice.backend}/{choice.device} in {load_ms:.1f} ms",
            flush=True,
        )
        return model, load_ms
    except Exception as exc:
        _last_runtime_error = str(exc)
        raise


def _release_cuda_cache() -> None:
    gc.collect()
    _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    """Return unused cached blocks to the driver, without gc.collect().

    After a successful pass its tensors are already freed by reference counting. A full gc.collect() in a
    process holding torch.compile state takes ~0.3-1 s, so it is reserved for failures and unloads.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _evict_model_variants(runtime_key: RuntimeKey) -> None:
    """Unload other precision/quantization/backend variants of the same model before loading a new one."""
    stale_keys = [key for key in _model_cache if key.model_id == runtime_key.model_id and key != runtime_key]
    if not stale_keys:
        return
    for key in stale_keys:
        _model_cache.pop(key, None)
    _release_cuda_cache()
    print(f"[G3_EMBED] Unloaded {len(stale_keys)} previous variant(s) of '{runtime_key.model_id}'", flush=True)


def _effective_max_seq_length(model_id: str, settings: dict[str, Any], native_max_seq_length: int) -> int | None:
    configured = int(settings.get("max_seq_length") or 0) or int(get_model_spec(model_id).get("default_max_seq_length") or 0)
    if configured <= 0:
        return native_max_seq_length or None
    if native_max_seq_length > 0:
        return min(configured, native_max_seq_length)
    return configured


def reported_max_seq_length(model_id: str) -> int | None:
    """The token window /embed applies to `model_id` (reported as max_model_len on /v1/models).

    A loaded model uses its native limit; an unloaded one the catalog limit, or None when the
    catalog does not know it (the native limit is only known after loading).
    """
    with settings_lock:
        settings_snapshot = dict(current_settings)
    native = 0
    for key, value in list(_model_cache.items()):
        if key.model_id == model_id:
            native = max(native, int(value.get("native_max_seq_length") or 0))
    if native <= 0:
        native = int(get_model_spec(model_id).get("max_seq_length") or 0)
        if native <= 0:
            return None
    return _effective_max_seq_length(model_id, settings_snapshot, native)


def _effective_encode_batch_size(model_id: str, settings: dict[str, Any], text_count: int) -> int:
    configured = int(settings.get("encode_batch_size") or 0)
    if configured <= 0:
        configured = int(get_model_spec(model_id).get("default_encode_batch_size") or settings.get("batch_max_texts") or 32)
    return max(1, min(configured, text_count))


def _encode_token_budget(model_id: str, settings: dict[str, Any], max_seq_length: int | None) -> int:
    """Padded tokens per forward pass: encode_batch_size sequences at the full window (the old worst case)."""
    window = int(max_seq_length or 0) or MAX_SEQ_LENGTH_LIMIT
    return _effective_encode_batch_size(model_id, settings, 1 << 30) * window


def plan_token_batches(lengths: Sequence[int], token_budget: int, max_rows: int = TOKEN_BATCH_MAX_ROWS) -> list[list[int]]:
    """Group input indices into forward passes of at most token_budget padded tokens (rows x longest).

    Longest first, so each pass is padded to its first entry and similar lengths share a pass: short texts
    go in passes of up to max_rows, full-window texts stay at token_budget // window.
    """
    order = sorted(range(len(lengths)), key=lambda index: -int(lengths[index]))
    batches: list[list[int]] = []
    start = 0
    while start < len(order):
        longest = max(1, int(lengths[order[start]]))
        rows = max(1, min(max_rows, token_budget // longest))
        batches.append(order[start : start + rows])
        start += rows
    return batches


def build_runtime_key(model_id: str, settings: dict[str, Any] | None = None) -> tuple[RuntimeKey, RuntimeChoice, dict[str, Any]]:
    with settings_lock:
        settings_snapshot = dict(current_settings)
    if settings:
        settings_snapshot.update(settings)

    normalized_model_id = (model_id or settings_snapshot.get("default_model") or "").strip()
    if not normalized_model_id:
        normalized_model_id = DEFAULT_MODEL_ID

    choice = resolve_runtime_choice(settings_snapshot)
    quantization = str(settings_snapshot.get("quantization") or "none")
    if not _is_cuda_pytorch(choice) or not get_model_spec(normalized_model_id).get("quantizable"):
        quantization = "none"
    cuda_graphs = bool(
        settings_snapshot.get("cuda_graphs", True)
        and _is_cuda_pytorch(choice)
        and get_model_spec(normalized_model_id).get("cuda_graphs")
    )
    return (
        RuntimeKey(
            model_id=normalized_model_id,
            backend=choice.backend,
            device=choice.device,
            precision=str(settings_snapshot.get("precision") or "auto"),
            compile_mode=str(settings_snapshot.get("compile_mode") or "off"),
            quantization=quantization,
            cuda_graphs=cuda_graphs,
        ),
        choice,
        settings_snapshot,
    )


class InputTooLongError(ValueError):
    """One or more inputs exceed the model's effective token window."""

    def __init__(self, model_id: str, max_tokens: int, too_long: list[dict[str, int]]):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.too_long = too_long
        super().__init__(
            f"{len(too_long)} input(s) exceed the token limit of {max_tokens} for '{model_id}'"
        )


@dataclass
class TokenCounts:
    model_id: str
    counts: list[int]
    max_tokens: int | None
    load_duration_ms: float


def _wrap_texts(texts: Sequence[str]) -> list[dict[str, str]]:
    # A bare string that looks like a media URL, path or data URI is loaded as media by sentence-transformers
    # (SSRF / local file read). Wrapped, it is always text; same token ids and embeddings as the bare string.
    return [{"text": text} for text in texts]


def _as_rows(vectors: Any) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
    return vectors.reshape(1, -1) if vectors.ndim == 1 else vectors


# Allocation failures outside the caching allocator are plain RuntimeErrors with these messages.
_CUDA_OOM_MESSAGES = ("CUDA out of memory", "CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_ALLOC_FAILED")


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(exc)
    return any(text in message for text in _CUDA_OOM_MESSAGES)


def _drop_tracebacks(exc: BaseException) -> None:
    """Free the frames (and their GPU tensors) a caught exception and its chain still reference."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current.__traceback__ = None
        current = current.__cause__ or current.__context__


def _check_cuda_headroom(required_mb: float, device: str) -> None:
    """Raise GpuMemoryUnavailable unless free + cached-unused memory covers required_mb plus the margin."""
    import torch

    cuda_device = torch.device(device)
    free_bytes, _ = torch.cuda.mem_get_info(cuda_device)
    cached_unused = torch.cuda.memory_reserved(cuda_device) - torch.cuda.memory_allocated(cuda_device)
    available_mb = (free_bytes + cached_unused) / (1024 * 1024)
    if available_mb < required_mb + CUDA_MEDIA_MARGIN_MB:
        raise GpuMemoryUnavailable(required_mb + CUDA_MEDIA_MARGIN_MB, available_mb)


def _load_entry(model_id: str, settings: dict[str, Any] | None):
    runtime_key, choice, settings_snapshot = build_runtime_key(model_id, settings)
    with _model_cache_lock:
        model, load_ms = _load_model(runtime_key, settings_snapshot, choice)
        cache_entry = _model_cache[runtime_key]
    max_seq_length = _effective_max_seq_length(
        runtime_key.model_id,
        settings_snapshot,
        int(cache_entry.get("native_max_seq_length") or 0),
    )
    return runtime_key, choice, settings_snapshot, model, load_ms, cache_entry, max_seq_length


def resolve_backend(model_id: str, settings: dict[str, Any] | None = None) -> str:
    """Backend ("pytorch" / "openvino") a request for this model would run on, without loading it."""
    _, choice, _ = build_runtime_key(model_id, settings)
    return choice.backend


def _serialize_preprocess(model: Any, tokenizer_lock: Any) -> None:
    """Run sentence-transformers' preprocess (tokenization) of every encode() pass with tokenizer_lock held.

    The shared (Rust-backed) tokenizer stores truncation/padding settings and every call re-applies its own,
    so token counting (no truncation) and encode() (truncated to max_seq_length) must not overlap. One lock
    around both lets counting use the model's own tokenizer: identical counts, no private copy (a deepcopy
    was 35-250 MB of RAM per model and 0.7-1.9 s on the first request), and counting only waits for a
    pass's tokenization, never for the GPU forward pass.
    """
    original_preprocess = model.preprocess

    def preprocess(*args: Any, **kwargs: Any) -> Any:
        with tokenizer_lock:
            return original_preprocess(*args, **kwargs)

    model.preprocess = preprocess


def _encode_text_rows(
    model: Any,
    model_id: str,
    settings: dict[str, Any],
    cache_entry: dict[str, Any],
    texts: list[str],
    max_seq_length: int | None,
) -> np.ndarray:
    """Normalized embeddings of texts in input order; call with encode_lock held.

    With encode_token_batching (default) passes are packed by padded tokens (plan_token_batches) instead of a
    fixed encode_batch_size texts, so many short texts share one pass while VRAM stays bounded by the old
    worst case (encode_batch_size full-window sequences).
    """
    batch_size = _effective_encode_batch_size(model_id, settings, len(texts))
    tokenizer = getattr(model, "tokenizer", None)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if not settings.get("encode_token_batching", True) or len(texts) <= batch_size or tokenizer is None:
        return _as_rows(
            model.encode(
                _wrap_texts(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
    with cache_entry["tokenizer_lock"]:
        token_ids = tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
    limit = int(max_seq_length or 0) or MAX_SEQ_LENGTH_LIMIT
    lengths = [min(len(ids), limit) for ids in token_ids]
    rows: list[Any] = [None] * len(texts)
    for batch in plan_token_batches(lengths, _encode_token_budget(model_id, settings, max_seq_length)):
        # batch_size = pass size, so sentence-transformers runs exactly this pass.
        vectors = model.encode(
            _wrap_texts([texts[index] for index in batch]),
            batch_size=len(batch),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for index, row in zip(batch, _as_rows(vectors)):
            rows[index] = row
    return np.stack(rows)


def count_input_tokens(model_id: str, inputs: Sequence[Any], settings: dict[str, Any] | None = None) -> TokenCounts:
    """Sequence length per input (prompt, special and media tokens included), untruncated.

    Inputs are strings or DecodedMedia. Loads the model if needed (load time in the result).
    """
    runtime_key, _, _, model, load_ms, cache_entry, max_tokens = _load_entry(model_id, settings)
    tokenizer = getattr(model.tokenizer, "tokenizer", model.tokenizer)
    if tokenizer is None:
        return TokenCounts(runtime_key.model_id, [], max_tokens, load_ms)
    texts = [str(item or "") for item in inputs if not isinstance(item, DecodedMedia)]
    media_texts = [item.text or "" for item in inputs if isinstance(item, DecodedMedia)]
    with cache_entry["tokenizer_lock"]:
        text_ids = (
            tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)["input_ids"] if texts else []
        )
        # Media inputs: text part without special tokens; input_tokens() adds them once per input.
        media_ids = (
            tokenizer(media_texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            if media_texts
            else []
        )
    text_iter, media_iter = iter(text_ids), iter(media_ids)
    counts = [
        input_tokens(len(next(media_iter)), item) if isinstance(item, DecodedMedia) else len(next(text_iter))
        for item in inputs
    ]
    return TokenCounts(runtime_key.model_id, counts, max_tokens, load_ms)


# Tokenizers of models that are not loaded, for rejecting over-limit texts without loading the model
# (CPU only, a few MB each). Least recently used first; model ids are client-controlled, so the cache is bounded.
_STANDALONE_TOKENIZERS_MAX = 4
_standalone_tokenizers: OrderedDict[tuple[str, str], Any] = OrderedDict()
_standalone_tokenizer_lock = threading.Lock()


def _load_standalone_tokenizer(model_id: str, cache_folder: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_folder or None, token=_resolve_token())


def _standalone_tokenizer(model_id: str, settings: dict[str, Any]) -> Any:
    """The model's tokenizer without its weights; call with _standalone_tokenizer_lock held. None if unavailable."""
    key = (model_id, resolve_model_cache_path(settings.get("model_cache_path", "")))
    if key in _standalone_tokenizers:
        _standalone_tokenizers.move_to_end(key)
        return _standalone_tokenizers[key]
    try:
        tokenizer = _load_standalone_tokenizer(*key)
    except Exception as exc:
        print(f"[G3_EMBED] Tokenizer for '{model_id}' unavailable, its token check loads the model: {exc}", flush=True)
        tokenizer = None
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    _standalone_tokenizers[key] = tokenizer
    while len(_standalone_tokenizers) > _STANDALONE_TOKENIZERS_MAX:
        _standalone_tokenizers.popitem(last=False)
    return tokenizer


def reject_too_long_before_load(model_id: str, texts: Sequence[str], settings: dict[str, Any] | None = None) -> None:
    """Raise InputTooLongError, counted with the tokenizer alone (CPU), while the model is not loaded.

    Without this, a request that is rejected anyway first loads (and compiles) its model onto the GPU. Only
    for models whose catalog entry has the native window (max_seq_length), so window, counts and error match
    what count_input_tokens reports after loading. Returns when the model is loaded, the window is unknown,
    the tokenizer is unavailable or every text fits; the caller then counts as before (loading the model).
    """
    runtime_key, _, settings_snapshot = build_runtime_key(model_id, settings)
    if runtime_key in _model_cache or not texts:
        return
    native = int(get_model_spec(runtime_key.model_id).get("max_seq_length") or 0)
    max_tokens = _effective_max_seq_length(runtime_key.model_id, settings_snapshot, native) if native > 0 else None
    if not max_tokens:
        return
    with _standalone_tokenizer_lock:
        tokenizer = _standalone_tokenizer(runtime_key.model_id, settings_snapshot)
        if tokenizer is None:
            return
        token_ids = tokenizer(
            [str(text or "") for text in texts], add_special_tokens=True, truncation=False, verbose=False
        )["input_ids"]
    too_long = [{"index": index, "tokens": len(ids)} for index, ids in enumerate(token_ids) if len(ids) > max_tokens]
    if too_long:
        raise InputTooLongError(runtime_key.model_id, max_tokens, too_long)


def validate_input_lengths(model_id: str, texts: list[str], settings: dict[str, Any] | None = None) -> float:
    """Raise InputTooLongError for texts longer than the effective window (prompt and special tokens included).

    Loads the model if needed (not when the tokenizer alone already rejects a text) and returns the load
    time in ms. Call it before batching so one oversized text only fails its own request.
    """
    reject_too_long_before_load(model_id, texts, settings)
    result = count_input_tokens(model_id, texts, settings)
    if not result.max_tokens:
        return result.load_duration_ms
    too_long = [
        {"index": index, "tokens": tokens}
        for index, tokens in enumerate(result.counts)
        if tokens > result.max_tokens
    ]
    if too_long:
        raise InputTooLongError(result.model_id, result.max_tokens, too_long)
    return result.load_duration_ms


def encode_texts(model_id: str, texts: list[str], settings: dict[str, Any] | None = None) -> EncodeResult:
    request_started = time.perf_counter()
    normalized_texts = [str(text or "") for text in texts]
    if not normalized_texts:
        normalized_texts = [""]

    runtime_key, choice, settings_snapshot, model, load_ms, cache_entry, max_seq_length = _load_entry(model_id, settings)

    out_of_memory = False
    # max_seq_length lives on the shared model and is read per mini-batch, so hold it fixed for the whole call.
    with cache_entry["encode_lock"]:
        if max_seq_length:
            model.max_seq_length = max_seq_length
        encode_started = time.perf_counter()
        try:
            vectors_np = _encode_text_rows(
                model, runtime_key.model_id, settings_snapshot, cache_entry, normalized_texts, max_seq_length
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            out_of_memory = True
        if out_of_memory:
            # Outside the except block, so the traceback no longer pins the failed pass's tensors.
            _release_cuda_cache()
            raise GpuOutOfMemory(runtime_key.model_id)
        encode_ms = (time.perf_counter() - encode_started) * 1000
    total_ms = (time.perf_counter() - request_started) * 1000

    vectors = _as_rows(vectors_np).astype(float).tolist()
    dimension = len(vectors[0]) if vectors else int(get_model_spec(runtime_key.model_id).get("dimensions") or 0)
    texts_per_second = len(normalized_texts) / (encode_ms / 1000) if encode_ms > 0 else None

    return EncodeResult(
        model_id=runtime_key.model_id,
        dimension=dimension,
        vectors=vectors,
        backend=choice.backend,
        device=choice.device,
        device_label=choice.device_label,
        load_duration_ms=round(load_ms, 3),
        encode_duration_ms=round(encode_ms, 3),
        total_duration_ms=round(total_ms, 3),
        texts_per_second=round(texts_per_second, 3) if texts_per_second is not None else None,
        input_count=len(normalized_texts),
        input_chars=sum(len(text) for text in normalized_texts),
    )


def encode_inputs(
    model_id: str,
    inputs: Sequence[Any],
    settings: dict[str, Any] | None = None,
    token_counts: Sequence[int] | None = None,
) -> EncodeResult:
    """Encode a mix of strings and DecodedMedia in one call, vectors in input order.

    Strings use the text path (encode_batch_size). Media inputs run in order-preserving chunks whose estimated
    extra VRAM stays within media_pass_budget_mb (token_counts from count_input_tokens size the LM cost). On CUDA
    each chunk needs free memory for its estimate plus a margin, and the cache is released afterwards.
    """
    request_started = time.perf_counter()
    if not inputs:
        raise ValueError("inputs must not be empty")
    runtime_key, choice, settings_snapshot, model, load_ms, cache_entry, max_seq_length = _load_entry(model_id, settings)
    text_positions = [index for index, item in enumerate(inputs) if not isinstance(item, DecodedMedia)]
    media_positions = [index for index, item in enumerate(inputs) if isinstance(item, DecodedMedia)]
    tokens = list(token_counts) if token_counts is not None else [0] * len(inputs)
    cost_mb = get_model_spec(runtime_key.model_id).get("media_cost_mb") or {}
    costs = [item_cost(inputs[index], int(tokens[index] or 0)) for index in media_positions]
    chunks = plan_chunks(costs, MediaLimits.from_settings(settings_snapshot).media_pass_budget_mb, cost_mb)
    on_cuda = _is_cuda_pytorch(choice)

    rows: list[Any] = [None] * len(inputs)
    failure: Exception | None = None
    with cache_entry["encode_lock"]:
        if max_seq_length:
            model.max_seq_length = max_seq_length
        encode_started = time.perf_counter()
        try:
            if text_positions:
                texts = [str(inputs[index] or "") for index in text_positions]
                vectors = _encode_text_rows(
                    model, runtime_key.model_id, settings_snapshot, cache_entry, texts, max_seq_length
                )
                for index, row in zip(text_positions, vectors):
                    rows[index] = row
            for chunk in chunks:
                positions = [media_positions[offset] for offset in chunk]
                if on_cuda:
                    _check_cuda_headroom(estimate_pass_mb([costs[offset] for offset in chunk], cost_mb), choice.device)
                # batch_size = chunk size, so sentence-transformers never regroups across our chunks.
                vectors = model.encode(
                    [inputs[index].to_st_input() for index in positions],
                    batch_size=len(positions),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for index, row in zip(positions, _as_rows(vectors)):
                    rows[index] = row
        except Exception as exc:
            failure = exc
        # Outside the except block: while an exception propagates, its traceback pins the failed pass's tensors.
        out_of_memory = failure is not None and _is_cuda_oom(failure)
        if failure is not None and (out_of_memory or (on_cuda and media_positions)):
            _drop_tracebacks(failure)
            # Hand reserved memory back to the other services sharing the GPU.
            _release_cuda_cache()
        elif on_cuda and media_positions:
            # Success: the pass's tensors are already freed; skip the slow gc.collect().
            _empty_cuda_cache()
        if out_of_memory:
            raise GpuOutOfMemory(runtime_key.model_id)
        if failure is not None:
            raise failure
        encode_ms = (time.perf_counter() - encode_started) * 1000
    total_ms = (time.perf_counter() - request_started) * 1000

    vectors_list = [np.asarray(row, dtype=np.float32).astype(float).tolist() for row in rows]
    dimension = len(vectors_list[0]) if vectors_list else int(get_model_spec(runtime_key.model_id).get("dimensions") or 0)
    per_second = len(inputs) / (encode_ms / 1000) if encode_ms > 0 else None
    return EncodeResult(
        model_id=runtime_key.model_id,
        dimension=dimension,
        vectors=vectors_list,
        backend=choice.backend,
        device=choice.device,
        device_label=choice.device_label,
        load_duration_ms=round(load_ms, 3),
        encode_duration_ms=round(encode_ms, 3),
        total_duration_ms=round(total_ms, 3),
        texts_per_second=round(per_second, 3) if per_second is not None else None,
        input_count=len(inputs),
        input_chars=sum(
            len(item.text or "") if isinstance(item, DecodedMedia) else len(str(item or "")) for item in inputs
        ),
        input_tokens=sum(int(count or 0) for count in token_counts) if token_counts is not None else None,
        media=media_summary(inputs),
    )


def compare_vectors(vector_a: list[float], vector_b: list[float]) -> dict[str, float | None]:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return {"cosine_similarity": None, "dot_product": None, "euclidean_distance": None}
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    dot = float(np.dot(a, b))
    norm_product = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = dot / norm_product if norm_product > 0 else 0.0
    distance = float(math.sqrt(float(np.sum((a - b) ** 2))))
    return {
        "cosine_similarity": round(cosine, 6),
        "dot_product": round(dot, 6),
        "euclidean_distance": round(distance, 6),
    }


def get_runtime_status() -> dict[str, Any]:
    # No _model_cache_lock: it is held for a whole load + compile warmup (minutes in "full" mode) and this
    # runs on the event loop for /health and admin pages. Copying the dict items is atomic under the GIL.
    loaded_models = [
        {
            "model_id": key.model_id,
            "backend": key.backend,
            "device": key.device,
            "precision": key.precision,
            "compile_mode": key.compile_mode,
            "quantization": key.quantization,
            "cuda_graphs": value["cuda_graphs"].status() if value.get("cuda_graphs") is not None else None,
            "max_seq_length": getattr(value.get("model"), "max_seq_length", None),
            "loaded_at": value.get("loaded_at"),
            "load_duration_ms": value.get("load_duration_ms"),
        }
        for key, value in list(_model_cache.items())
    ]
    with settings_lock:
        settings_snapshot = dict(current_settings)
    choice = resolve_runtime_choice(settings_snapshot)
    return {
        "loaded_models": loaded_models,
        "last_runtime_error": _last_runtime_error,
        "resolved_runtime": {
            "backend": choice.backend,
            "device": choice.device,
            "device_label": choice.device_label,
            "source": choice.source,
            "detail": choice.detail,
        },
        "system": get_system_snapshot(),
    }
