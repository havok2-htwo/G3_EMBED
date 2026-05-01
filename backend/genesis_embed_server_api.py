from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .genesis_embed_server_globals import MODEL_OPTIONS, batch_history, current_settings, embedding_history, history_lock, settings_lock
from .genesis_embed_server_runtime import EncodeResult, encode_texts, get_runtime_status
from .genesis_embed_server_storage import log_embedding


class EmbedRequest(BaseModel):
    model: str | None = Field(default=None, description="SentenceTransformer model id")
    inputs: Union[str, List[str]]


class EmbedResponse(BaseModel):
    model_id: str
    dimension: int
    vectors: List[List[float]]
    backend: str
    device: str
    device_label: str
    input_count: int
    input_chars: int
    load_duration_ms: float
    encode_duration_ms: float
    total_duration_ms: float
    texts_per_second: float | None = None
    batched: bool = False


class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str | None = None
    encoding_format: str | None = None
    user: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_inputs(inputs: Union[str, List[str]]) -> list[str]:
    normalized = inputs if isinstance(inputs, list) else [inputs]
    return [str(item or "") for item in normalized]


def _result_to_response(result: EncodeResult, *, batched: bool) -> dict[str, Any]:
    return {
        "model_id": result.model_id,
        "dimension": result.dimension,
        "vectors": result.vectors,
        "backend": result.backend,
        "device": result.device,
        "device_label": result.device_label,
        "input_count": result.input_count,
        "input_chars": result.input_chars,
        "load_duration_ms": result.load_duration_ms,
        "encode_duration_ms": result.encode_duration_ms,
        "total_duration_ms": result.total_duration_ms,
        "texts_per_second": result.texts_per_second,
        "batched": batched,
    }


def _record_history(entry: dict[str, Any]) -> None:
    with history_lock:
        embedding_history.appendleft(entry)
    log_embedding(entry)


async def embed_texts_for_request(
    request: Request,
    *,
    model_id: str | None,
    inputs: list[str],
    route: str,
    force_direct: bool = False,
) -> tuple[EncodeResult, bool]:
    if not inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")

    with settings_lock:
        settings_snapshot = dict(current_settings)
    effective_model = (model_id or settings_snapshot.get("default_model") or "").strip()
    if not effective_model:
        raise HTTPException(status_code=400, detail="model is required")

    source_ip = request.client.host if request.client else "unknown"
    batched = bool(settings_snapshot.get("batching_enabled")) and not force_direct
    try:
        if batched and getattr(request.app.state, "embed_batch_manager", None) is not None:
            result = await request.app.state.embed_batch_manager.enqueue(effective_model, inputs, settings_snapshot)
        else:
            result = await asyncio.to_thread(encode_texts, effective_model, inputs, settings_snapshot)
            batched = False
    except Exception as exc:
        entry = {
            "timestamp": _now_iso(),
            "source_ip": source_ip,
            "route": route,
            "model_id": effective_model,
            "input_count": len(inputs),
            "input_chars": sum(len(text) for text in inputs),
            "success": False,
            "error": str(exc),
            "batched": batched,
        }
        _record_history(entry)
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc

    entry = {
        "timestamp": _now_iso(),
        "source_ip": source_ip,
        "route": route,
        "model_id": result.model_id,
        "dimension": result.dimension,
        "backend": result.backend,
        "device": result.device,
        "device_label": result.device_label,
        "input_count": result.input_count,
        "input_chars": result.input_chars,
        "load_duration_ms": result.load_duration_ms,
        "encode_duration_ms": result.encode_duration_ms,
        "total_duration_ms": result.total_duration_ms,
        "texts_per_second": result.texts_per_second,
        "success": True,
        "batched": batched,
    }
    _record_history(entry)
    return result, batched


def create_api(app: FastAPI) -> FastAPI:
    @app.get("/health")
    async def health():
        with settings_lock:
            settings_snapshot = dict(current_settings)
        runtime = get_runtime_status()
        return {
            "ok": True,
            "service": "G3_EMBED",
            "default_model": settings_snapshot.get("default_model"),
            "settings": {
                "execution_target": settings_snapshot.get("execution_target"),
                "backend_override": settings_snapshot.get("backend_override"),
                "precision": settings_snapshot.get("precision"),
                "compile_mode": settings_snapshot.get("compile_mode"),
                "batching_enabled": settings_snapshot.get("batching_enabled"),
            },
            **runtime,
        }

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: Request, payload: EmbedRequest):
        result, batched = await embed_texts_for_request(
            request,
            model_id=payload.model,
            inputs=_normalize_inputs(payload.inputs),
            route="/embed",
        )
        return _result_to_response(result, batched=batched)

    @app.get("/v1/models")
    async def v1_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": option["value"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "g3_embed",
                }
                for option in MODEL_OPTIONS
            ],
        }

    @app.post("/v1/embeddings")
    async def v1_embeddings(request: Request, payload: OpenAIEmbeddingRequest):
        result, _ = await embed_texts_for_request(
            request,
            model_id=payload.model,
            inputs=_normalize_inputs(payload.input),
            route="/v1/embeddings",
        )
        approximate_tokens = max(1, result.input_chars // 4)
        return {
            "object": "list",
            "model": result.model_id,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector,
                }
                for index, vector in enumerate(result.vectors)
            ],
            "usage": {
                "prompt_tokens": approximate_tokens,
                "total_tokens": approximate_tokens,
            },
        }

    return app
