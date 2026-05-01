from __future__ import annotations

import asyncio
from statistics import mean
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from .genesis_embed_server_api import embed_texts_for_request
from .genesis_embed_server_auth import get_admin_key_store, require_admin
from .genesis_embed_server_globals import (
    BACKEND_OPTIONS,
    COMPILE_OPTIONS,
    EXECUTION_TARGET_OPTIONS,
    MODEL_OPTIONS,
    PRECISION_OPTIONS,
    batch_history,
    current_settings,
    embedding_history,
    history_lock,
    settings_lock,
)
from .genesis_embed_server_hardware import get_system_snapshot
from .genesis_embed_server_model_manager import delete_model_cache, list_model_statuses, queue_model_download
from .genesis_embed_server_runtime import compare_vectors, encode_texts, get_runtime_status
from .genesis_embed_server_storage import normalize_settings, save_settings


class AdminSettingsPayload(BaseModel):
    default_model: str
    execution_target: str
    backend_override: str
    model_cache_path: str
    batching_enabled: bool
    batch_wait_time_ms: int
    batch_max_texts: int
    batch_max_chars: int
    precision: str
    compile_mode: str
    warmup_on_load: bool
    huggingface_token: str


class AdminModelActionPayload(BaseModel):
    model_id: str
    storage_path: str | None = None
    huggingface_token: str | None = None


class ComparePayload(BaseModel):
    model: str | None = None
    text_a: str
    text_b: str


class BenchmarkPayload(BaseModel):
    model: str | None = None
    inputs: List[str]
    repeat_count: int = 1


def _serialize_settings() -> Dict[str, Any]:
    with settings_lock:
        return normalize_settings(dict(current_settings))


def _settings_options() -> Dict[str, List[Dict[str, str]]]:
    return {
        "models": MODEL_OPTIONS,
        "execution_targets": EXECUTION_TARGET_OPTIONS,
        "backends": BACKEND_OPTIONS,
        "precisions": PRECISION_OPTIONS,
        "compile_modes": COMPILE_OPTIONS,
    }


def _effective_model_storage_path(storage_path: str | None = None) -> str:
    if storage_path is not None:
        return str(storage_path)
    return str(_serialize_settings().get("model_cache_path", ""))


def create_admin_api(app: FastAPI) -> FastAPI:
    @app.get("/api/admin/keys")
    async def admin_get_keys(_: dict[str, str] = Depends(require_admin)):
        return get_admin_key_store().list_keys()

    @app.post("/api/admin/keys")
    async def admin_rotate_key(_: dict[str, str] = Depends(require_admin)):
        return {
            "key": get_admin_key_store().rotate_admin_key(),
            "keys": get_admin_key_store().list_keys(),
        }

    @app.get("/api/admin/settings")
    async def admin_get_settings(_: dict[str, str] = Depends(require_admin)):
        settings_snapshot = _serialize_settings()
        return {
            "settings": settings_snapshot,
            "options": _settings_options(),
            "models": list_model_statuses(settings_snapshot.get("model_cache_path", "")),
            "runtime": get_runtime_status(),
        }

    @app.put("/api/admin/settings")
    async def admin_update_settings(payload: AdminSettingsPayload, _: dict[str, str] = Depends(require_admin)):
        payload_data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        normalized = normalize_settings(payload_data)
        with settings_lock:
            current_settings.clear()
            current_settings.update(save_settings(normalized))
        return {
            "ok": True,
            "settings": _serialize_settings(),
            "options": _settings_options(),
            "models": list_model_statuses(normalized.get("model_cache_path", "")),
            "runtime": get_runtime_status(),
        }

    @app.get("/api/admin/system")
    async def admin_system(_: dict[str, str] = Depends(require_admin)):
        return {"system": get_system_snapshot(), "runtime": get_runtime_status()}

    @app.get("/api/admin/models")
    async def admin_get_models(storage_path: str | None = None, _: dict[str, str] = Depends(require_admin)):
        return {"models": list_model_statuses(_effective_model_storage_path(storage_path))}

    @app.post("/api/admin/models/download")
    async def admin_download_model(payload: AdminModelActionPayload, _: dict[str, str] = Depends(require_admin)):
        try:
            job = queue_model_download(
                payload.model_id,
                _effective_model_storage_path(payload.storage_path),
                payload.huggingface_token,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "job": job,
            "models": list_model_statuses(_effective_model_storage_path(payload.storage_path)),
        }

    @app.post("/api/admin/models/delete")
    async def admin_delete_model(payload: AdminModelActionPayload, _: dict[str, str] = Depends(require_admin)):
        try:
            result = delete_model_cache(payload.model_id, _effective_model_storage_path(payload.storage_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {**result, "models": list_model_statuses(_effective_model_storage_path(payload.storage_path))}

    @app.get("/api/admin/stats")
    async def admin_stats(_: dict[str, str] = Depends(require_admin)):
        with history_lock:
            history_items = list(embedding_history)
        durations = [entry.get("total_duration_ms") for entry in history_items if entry.get("total_duration_ms") is not None]
        encode_durations = [entry.get("encode_duration_ms") for entry in history_items if entry.get("encode_duration_ms") is not None]
        text_counts = [entry.get("input_count") for entry in history_items if entry.get("input_count") is not None]
        return {
            "summary": {
                "total_requests": len(history_items),
                "successful_requests": sum(1 for entry in history_items if entry.get("success")),
                "avg_total_duration_ms": round(mean(durations), 2) if durations else None,
                "avg_encode_duration_ms": round(mean(encode_durations), 2) if encode_durations else None,
                "total_texts": sum(int(value or 0) for value in text_counts),
            },
            "history": history_items[:50],
            "recent_batches": list(batch_history),
        }

    @app.get("/api/admin/queue")
    async def admin_queue(request: Request, _: dict[str, str] = Depends(require_admin)):
        manager = getattr(request.app.state, "embed_batch_manager", None)
        if manager is None:
            return {"worker_running": False, "recent_batches": list(batch_history)}
        return manager.snapshot()

    @app.post("/api/admin/compare")
    async def admin_compare(
        request: Request,
        payload: ComparePayload,
        _: dict[str, str] = Depends(require_admin),
    ):
        result, batched = await embed_texts_for_request(
            request,
            model_id=payload.model,
            inputs=[payload.text_a, payload.text_b],
            route="/api/admin/compare",
        )
        metrics = compare_vectors(result.vectors[0], result.vectors[1]) if len(result.vectors) >= 2 else {}
        return {
            "ok": True,
            "model_id": result.model_id,
            "dimension": result.dimension,
            "backend": result.backend,
            "device": result.device,
            "device_label": result.device_label,
            "batched": batched,
            "load_duration_ms": result.load_duration_ms,
            "encode_duration_ms": result.encode_duration_ms,
            "total_duration_ms": result.total_duration_ms,
            "vectors": result.vectors,
            **metrics,
        }

    @app.post("/api/admin/benchmark")
    async def admin_benchmark(payload: BenchmarkPayload, _: dict[str, str] = Depends(require_admin)):
        repeat_count = max(1, min(128, int(payload.repeat_count or 1)))
        inputs = [str(item or "") for item in payload.inputs]
        if not inputs:
            raise HTTPException(status_code=400, detail="inputs must not be empty")
        model_id = payload.model or _serialize_settings().get("default_model")
        started = asyncio.get_running_loop().time()
        last_result = None
        for _ in range(repeat_count):
            last_result = await asyncio.to_thread(encode_texts, model_id, inputs)
        total_ms = round((asyncio.get_running_loop().time() - started) * 1000, 3)
        assert last_result is not None
        total_texts = len(inputs) * repeat_count
        return {
            "ok": True,
            "model_id": last_result.model_id,
            "dimension": last_result.dimension,
            "backend": last_result.backend,
            "device": last_result.device,
            "device_label": last_result.device_label,
            "repeat_count": repeat_count,
            "texts_per_run": len(inputs),
            "total_texts": total_texts,
            "total_chars": sum(len(text) for text in inputs) * repeat_count,
            "total_wall_time_ms": total_ms,
            "avg_wall_time_per_run_ms": round(total_ms / repeat_count, 3),
            "texts_per_second": round(total_texts / (total_ms / 1000), 3) if total_ms > 0 else None,
        }

    return app
