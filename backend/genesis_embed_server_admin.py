from __future__ import annotations

import asyncio
from statistics import mean
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .genesis_embed_server_api import embed_texts_for_request
from .g3_gpu import perf_payload
from .genesis_embed_server_auth import (
    ALIAS_MAX_LENGTH,
    DEFAULT_ADMIN_USERNAME,
    SESSION_COOKIE_NAME,
    clear_session_cookie,
    client_key,
    get_auth_store,
    login_rate_limiter,
    require_admin,
    require_session,
    set_session_cookie,
)
from .genesis_embed_server_globals import (
    BACKEND_OPTIONS,
    COMPILE_OPTIONS,
    EXECUTION_TARGET_OPTIONS,
    MAX_INPUTS_PER_REQUEST,
    MODEL_OPTIONS,
    PRECISION_OPTIONS,
    QUANTIZATION_OPTIONS,
    TOO_LONG_POLICY_OPTIONS,
    batch_history,
    current_settings,
    embedding_history,
    history_lock,
    request_counters,
    settings_lock,
)
from .genesis_embed_server_hardware import get_system_snapshot, gpu_device_options
from .genesis_embed_server_model_manager import delete_model_cache, list_model_statuses, queue_model_download
from .genesis_embed_server_presets import delete_user_preset, resolve_presets, save_user_preset
from .genesis_embed_server_runtime import compare_vectors, encode_texts, get_runtime_status, texts_rate
from .genesis_embed_server_storage import normalize_settings, save_settings


class AdminSettingsPayload(BaseModel):
    # Every field is optional: an omitted (or null) field keeps its current value (see PUT below), so a client
    # never has to invent a value for a field the user left empty.
    default_model: Optional[str] = None
    execution_target: Optional[str] = None
    gpu_device: Optional[str] = None
    backend_override: Optional[str] = None
    model_cache_path: Optional[str] = None
    batching_enabled: Optional[bool] = None
    batch_wait_time_ms: Optional[int] = None
    batch_max_texts: Optional[int] = None
    batch_max_chars: Optional[int] = None
    encode_batch_size: int = 0
    encode_token_batching: bool = True
    max_seq_length: int = 0
    too_long_policy: str = "reject"
    precision: Optional[str] = None
    quantization: str = "none"
    compile_mode: Optional[str] = None
    cuda_graphs: bool = True
    warmup_on_load: Optional[bool] = None
    idle_trim_seconds: int = 5
    # Write-only: empty, missing or the masked value from GET keeps the stored token; clear_huggingface_token
    # removes it.
    huggingface_token: Optional[str] = None
    clear_huggingface_token: bool = False
    # Optional so older frontends keep working; an omitted field keeps its current value (see PUT below).
    image_inputs_enabled: bool = True
    audio_inputs_enabled: bool = True
    video_inputs_enabled: bool = True
    max_images_per_input: int = 4
    max_images_per_request: int = 16
    max_audio_clips_per_input: int = 2
    max_audio_seconds_per_clip: int = 30
    max_audio_seconds_per_request: int = 120
    max_videos_per_request: int = 1
    max_video_frames: int = 8
    max_media_item_mb: int = 8
    max_image_megapixels: int = 25
    max_request_mb: int = 32
    media_pass_budget_mb: int = 1200
    max_concurrent_media_requests: int = 2


class PresetPayload(BaseModel):
    # Settings values; anything outside presets.PRESET_KEYS is dropped, the rest normalised like a saved setting.
    values: Dict[str, Any] = Field(default_factory=dict)


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
    inputs: List[str] = Field(max_length=MAX_INPUTS_PER_REQUEST)
    repeat_count: int = 1


class LoginPayload(BaseModel):
    username: str
    password: str


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str


class CreateApiKeyPayload(BaseModel):
    alias: str = Field(default="", max_length=ALIAS_MAX_LENGTH)


def mask_token(token: str) -> str:
    """"hf_…abcd": enough to recognise which token is stored, never enough to use it."""
    token = (token or "").strip()
    if not token:
        return ""
    head = token[:3] if token.startswith("hf_") else ""
    return f"{head}…{token[-4:]}" if len(token) > 8 else "…"


def _serialize_settings() -> Dict[str, Any]:
    with settings_lock:
        return normalize_settings(dict(current_settings))


def _public_settings() -> Dict[str, Any]:
    """Settings for the browser: the Hugging Face token only as a masked hint, never in clear text."""
    settings = _serialize_settings()
    settings["huggingface_token"] = mask_token(settings.get("huggingface_token", ""))
    return settings


def _token_fields() -> Dict[str, Any]:
    token = _serialize_settings().get("huggingface_token", "")
    return {"huggingface_token_set": bool(token), "huggingface_token_hint": mask_token(token) or None}


def _settings_options() -> Dict[str, List[Dict[str, str]]]:
    return {
        "models": MODEL_OPTIONS,
        "execution_targets": EXECUTION_TARGET_OPTIONS,
        "gpu_devices": gpu_device_options(_serialize_settings().get("gpu_device")),
        "backends": BACKEND_OPTIONS,
        "precisions": PRECISION_OPTIONS,
        "quantizations": QUANTIZATION_OPTIONS,
        "too_long_policies": TOO_LONG_POLICY_OPTIONS,
        "compile_modes": COMPILE_OPTIONS,
    }


def _effective_model_storage_path(storage_path: str | None = None) -> str:
    if storage_path is not None:
        return str(storage_path)
    return str(_serialize_settings().get("model_cache_path", ""))


def create_admin_api(app: FastAPI) -> FastAPI:
    # --- Auth: username/password login backed by an httpOnly session cookie ---
    @app.post("/api/admin/auth/login")
    async def admin_login(payload: LoginPayload, request: Request, response: Response):
        store = get_auth_store()
        store.ensure_usable()
        client = client_key(request)
        retry_after = login_rate_limiter.begin(client)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed logins. Try again in {retry_after} s.",
                headers={"Retry-After": str(retry_after)},
            )
        user = await asyncio.to_thread(store.verify_user, payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        login_rate_limiter.succeeded(client)
        token = store.create_session(user["username"])
        store.touch_login(user["username"])
        set_session_cookie(response, request, token)
        return {"username": user["username"], "must_change_password": bool(user.get("must_change_password"))}

    @app.get("/api/admin/auth/setup")
    async def admin_setup_status():
        # Public on purpose: the login page shows the default-credentials hint only while setup is pending.
        store = get_auth_store()
        required = not store.load_error and store.setup_required()
        return {"setup_required": required, "default_username": DEFAULT_ADMIN_USERNAME if required else None}

    @app.post("/api/admin/auth/logout")
    async def admin_logout(request: Request, response: Response, _: dict = Depends(require_session)):
        get_auth_store().delete_session(request.cookies.get(SESSION_COOKIE_NAME))
        clear_session_cookie(response)
        return {"ok": True}

    @app.get("/api/admin/auth/whoami")
    async def admin_whoami(ctx: dict = Depends(require_session)):
        return {
            "username": ctx["username"],
            "must_change_password": ctx["must_change_password"],
            "session_expires_at": ctx.get("expires_at"),
        }

    @app.post("/api/admin/auth/change-password")
    async def admin_change_password(
        payload: ChangePasswordPayload,
        request: Request,
        response: Response,
        ctx: dict = Depends(require_session),
    ):
        store = get_auth_store()
        if not await asyncio.to_thread(store.verify_user, ctx["username"], payload.current_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        await asyncio.to_thread(store.set_password, ctx["username"], payload.new_password)
        token = store.create_session(ctx["username"])
        set_session_cookie(response, request, token)
        return {"ok": True, "must_change_password": False}

    # --- Client API keys (admin-managed; used to authorize the public /embed API) ---
    @app.get("/api/admin/api-keys")
    async def admin_list_api_keys(_: dict = Depends(require_admin)):
        return {"keys": get_auth_store().list_api_keys()}

    @app.post("/api/admin/api-keys")
    async def admin_create_api_key(payload: CreateApiKeyPayload, _: dict = Depends(require_admin)):
        return get_auth_store().create_api_key(payload.alias)

    @app.delete("/api/admin/api-keys/{key_id}")
    async def admin_delete_api_key(key_id: str, _: dict = Depends(require_admin)):
        if not get_auth_store().delete_api_key(key_id):
            raise HTTPException(status_code=404, detail="API key not found.")
        return Response(status_code=204)

    @app.get("/api/admin/settings")
    async def admin_get_settings(_: dict[str, str] = Depends(require_admin)):
        settings_snapshot = _serialize_settings()
        return {
            "settings": _public_settings(),
            **_token_fields(),
            "options": _settings_options(),
            "models": list_model_statuses(settings_snapshot.get("model_cache_path", "")),
            "runtime": get_runtime_status(),
        }

    @app.put("/api/admin/settings")
    async def admin_update_settings(payload: AdminSettingsPayload, _: dict[str, str] = Depends(require_admin)):
        # Only the fields the client sent: an omitted optional field keeps its current value, so an older
        # client cannot re-enable a disabled modality or reset a lowered limit to its default.
        sent = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else payload.dict(exclude_unset=True)
        sent = {key: value for key, value in sent.items() if value is not None}
        clear_token = bool(sent.pop("clear_huggingface_token", False))
        with settings_lock:
            stored_token = str(current_settings.get("huggingface_token", "") or "")
            token = str(sent.pop("huggingface_token", "") or "").strip()
            if clear_token:
                sent["huggingface_token"] = ""
            elif token and token != mask_token(stored_token):
                sent["huggingface_token"] = token
            normalized = normalize_settings({**current_settings, **sent})
            current_settings.clear()
            current_settings.update(save_settings(normalized))
        return {
            "ok": True,
            "settings": _public_settings(),
            **_token_fields(),
            "options": _settings_options(),
            "models": list_model_statuses(normalized.get("model_cache_path", "")),
            "runtime": get_runtime_status(),
        }

    # --- Settings presets (built-in GPU tiers + saved ones; loading fills the form, Save Settings applies) ---
    @app.get("/api/admin/settings/presets")
    async def admin_list_presets(_: dict[str, str] = Depends(require_admin)):
        return {"presets": await asyncio.to_thread(resolve_presets)}

    @app.put("/api/admin/settings/presets/{name}")
    async def admin_save_preset(name: str, payload: PresetPayload, _: dict[str, str] = Depends(require_admin)):
        try:
            preset = await asyncio.to_thread(save_user_preset, name, payload.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "preset": preset}

    @app.delete("/api/admin/settings/presets/{name}")
    async def admin_delete_preset(name: str, _: dict[str, str] = Depends(require_admin)):
        try:
            deleted = await asyncio.to_thread(delete_user_preset, name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No saved preset '{name[:100]}'.")
        return {"ok": True}

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
            counters = dict(request_counters)
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
                "requests_since_start": counters["requests"],
                "errors_since_start": counters["errors"],
                "started_at": counters["started_at"],
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

    @app.get("/api/admin/perf")
    async def admin_perf(request: Request, _: dict[str, str] = Depends(require_admin)):
        manager = getattr(request.app.state, "embed_batch_manager", None)
        metrics = [
            {"key": "throughput", "label": "Texts/s", "unit": "texts/s", "value": round(texts_rate.rate(), 3), "window_s": 30},
            {
                "key": "queue",
                "label": "Queue depth",
                "unit": "",
                "value": manager.queue_depth() if manager is not None else 0,
                "decimals": 0,
                "integer": True,
            },
        ]
        # NVML / nvidia-smi can take ~100 ms: never on the event loop.
        return await asyncio.to_thread(perf_payload, "G3 Embed", metrics)

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
