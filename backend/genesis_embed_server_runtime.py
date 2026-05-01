from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .genesis_embed_server_globals import current_settings, get_model_spec, resolve_model_cache_path, settings_lock
from .genesis_embed_server_hardware import RuntimeChoice, get_system_snapshot, resolve_runtime_choice


@dataclass(frozen=True)
class RuntimeKey:
    model_id: str
    backend: str
    device: str
    precision: str
    compile_mode: str


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


_model_cache_lock = threading.RLock()
_model_cache: dict[RuntimeKey, dict[str, Any]] = {}
_last_runtime_error: str | None = None


def _resolve_token() -> str | None:
    with settings_lock:
        token = str(current_settings.get("huggingface_token", "")).strip()
    return token or None


def _model_kwargs_for_precision(choice: RuntimeChoice, precision: str) -> dict[str, Any]:
    if choice.backend != "pytorch" or choice.device == "cpu":
        return {}
    try:
        import torch
    except Exception:
        return {}

    selected = precision
    if selected == "auto":
        selected = "float16" if choice.device.startswith("cuda") else "float32"
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_by_name.get(selected)
    return {"torch_dtype": torch_dtype} if torch_dtype is not None else {}


def _maybe_compile_sentence_transformer(model: Any, compile_mode: str, choice: RuntimeChoice) -> Any:
    if compile_mode == "off" or choice.backend != "pytorch" or choice.device == "cpu":
        return model
    try:
        import torch

        if not hasattr(torch, "compile"):
            return model
        first_module = model._first_module()
        auto_model = getattr(first_module, "auto_model", None)
        if auto_model is None:
            return model
        first_module.auto_model = torch.compile(auto_model, mode=compile_mode)
    except Exception as exc:
        print(f"[G3_EMBED] torch.compile skipped: {exc}", flush=True)
    return model


def _load_model(runtime_key: RuntimeKey, settings: dict[str, Any], choice: RuntimeChoice) -> tuple[Any, float]:
    global _last_runtime_error

    cached = _model_cache.get(runtime_key)
    if cached is not None:
        return cached["model"], 0.0

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
            model_kwargs = _model_kwargs_for_precision(choice, runtime_key.precision)
            if model_kwargs:
                kwargs["model_kwargs"] = model_kwargs

        model = SentenceTransformer(runtime_key.model_id, **kwargs)
        model = _maybe_compile_sentence_transformer(model, runtime_key.compile_mode, choice)

        if settings.get("warmup_on_load"):
            model.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

        load_ms = (time.perf_counter() - load_started) * 1000
        _model_cache[runtime_key] = {
            "model": model,
            "loaded_at": time.time(),
            "load_duration_ms": load_ms,
            "choice": choice,
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


def build_runtime_key(model_id: str, settings: dict[str, Any] | None = None) -> tuple[RuntimeKey, RuntimeChoice, dict[str, Any]]:
    with settings_lock:
        settings_snapshot = dict(current_settings)
    if settings:
        settings_snapshot.update(settings)

    normalized_model_id = (model_id or settings_snapshot.get("default_model") or "").strip()
    if not normalized_model_id:
        normalized_model_id = "intfloat/multilingual-e5-small"

    choice = resolve_runtime_choice(settings_snapshot)
    return (
        RuntimeKey(
            model_id=normalized_model_id,
            backend=choice.backend,
            device=choice.device,
            precision=str(settings_snapshot.get("precision") or "auto"),
            compile_mode=str(settings_snapshot.get("compile_mode") or "off"),
        ),
        choice,
        settings_snapshot,
    )


def encode_texts(model_id: str, texts: list[str], settings: dict[str, Any] | None = None) -> EncodeResult:
    request_started = time.perf_counter()
    normalized_texts = [str(text or "") for text in texts]
    if not normalized_texts:
        normalized_texts = [""]

    runtime_key, choice, settings_snapshot = build_runtime_key(model_id, settings)
    with _model_cache_lock:
        model, load_ms = _load_model(runtime_key, settings_snapshot, choice)

    encode_started = time.perf_counter()
    vectors_np = model.encode(
        normalized_texts,
        batch_size=max(1, min(int(settings_snapshot.get("batch_max_texts") or 32), len(normalized_texts))),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    encode_ms = (time.perf_counter() - encode_started) * 1000
    total_ms = (time.perf_counter() - request_started) * 1000

    if not isinstance(vectors_np, np.ndarray):
        vectors_np = np.asarray(vectors_np, dtype=np.float32)
    if vectors_np.ndim == 1:
        vectors_np = vectors_np.reshape(1, -1)
    vectors = vectors_np.astype(float).tolist()
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
    with _model_cache_lock:
        loaded_models = [
            {
                "model_id": key.model_id,
                "backend": key.backend,
                "device": key.device,
                "precision": key.precision,
                "compile_mode": key.compile_mode,
                "loaded_at": value.get("loaded_at"),
                "load_duration_ms": value.get("load_duration_ms"),
            }
            for key, value in _model_cache.items()
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
