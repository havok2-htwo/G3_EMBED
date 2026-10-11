from __future__ import annotations

import copy
import gc
import logging
import math
import threading
import time
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


@dataclass(frozen=True)
class RuntimeKey:
    model_id: str
    backend: str
    device: str
    precision: str
    compile_mode: str
    quantization: str = "none"


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

        if compiled:
            _warmup_compiled_model(
                model,
                _effective_max_seq_length(runtime_key.model_id, settings, native_max_seq_length),
                _effective_encode_batch_size(runtime_key.model_id, settings, 2),
            )
        elif settings.get("warmup_on_load"):
            model.encode(_wrap_texts(["warmup"]), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

        load_ms = (time.perf_counter() - load_started) * 1000
        _model_cache[runtime_key] = {
            "model": model,
            "loaded_at": time.time(),
            "load_duration_ms": load_ms,
            "choice": choice,
            "native_max_seq_length": native_max_seq_length,
            "encode_lock": threading.Lock(),
            "count_lock": threading.Lock(),
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


def _effective_encode_batch_size(model_id: str, settings: dict[str, Any], text_count: int) -> int:
    configured = int(settings.get("encode_batch_size") or 0)
    if configured <= 0:
        configured = int(get_model_spec(model_id).get("default_encode_batch_size") or settings.get("batch_max_texts") or 32)
    return max(1, min(configured, text_count))


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
    return (
        RuntimeKey(
            model_id=normalized_model_id,
            backend=choice.backend,
            device=choice.device,
            precision=str(settings_snapshot.get("precision") or "auto"),
            compile_mode=str(settings_snapshot.get("compile_mode") or "off"),
            quantization=quantization,
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
    # A private copy: the shared (Rust-backed) tokenizer is mutated with truncation settings by a concurrent
    # model.encode and must not be called with different settings outside encode_lock.
    with cache_entry["count_lock"]:
        count_tokenizer = cache_entry.get("count_tokenizer")
        if count_tokenizer is None:
            count_tokenizer = copy.deepcopy(tokenizer)
            cache_entry["count_tokenizer"] = count_tokenizer
        text_ids = (
            count_tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)["input_ids"] if texts else []
        )
        # Media inputs: text part without special tokens; input_tokens() adds them once per input.
        media_ids = (
            count_tokenizer(media_texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            if media_texts
            else []
        )
    text_iter, media_iter = iter(text_ids), iter(media_ids)
    counts = [
        input_tokens(len(next(media_iter)), item) if isinstance(item, DecodedMedia) else len(next(text_iter))
        for item in inputs
    ]
    return TokenCounts(runtime_key.model_id, counts, max_tokens, load_ms)


def validate_input_lengths(model_id: str, texts: list[str], settings: dict[str, Any] | None = None) -> float:
    """Raise InputTooLongError for texts longer than the effective window (prompt and special tokens included).

    Loads the model if needed and returns the load time in ms. Call it before batching so one oversized
    text only fails its own request.
    """
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
            vectors_np = model.encode(
                _wrap_texts(normalized_texts),
                batch_size=_effective_encode_batch_size(runtime_key.model_id, settings_snapshot, len(normalized_texts)),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
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
                vectors = model.encode(
                    _wrap_texts(texts),
                    batch_size=_effective_encode_batch_size(runtime_key.model_id, settings_snapshot, len(texts)),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for index, row in zip(text_positions, _as_rows(vectors)):
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
        if out_of_memory or (on_cuda and media_positions):
            if failure is not None:
                _drop_tracebacks(failure)
            # Hand reserved memory back to the other services sharing the GPU.
            _release_cuda_cache()
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
