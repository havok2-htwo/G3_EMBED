from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, List, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .genesis_embed_server_auth import authorize_api_key, get_auth_store
from .genesis_embed_server_globals import (
    MAX_INPUTS_PER_REQUEST,
    MAX_MEDIA_ITEMS_PER_FIELD,
    MEDIA_LIMIT_SETTINGS,
    MODEL_OPTIONS,
    OTHER_ROUTES_MAX_BODY_MB,
    batch_history,
    current_settings,
    embedding_history,
    get_model_spec,
    history_lock,
    settings_lock,
)
from .genesis_embed_server_media import (
    DecodedMedia,
    InputCost,
    MediaInputData,
    MediaLimits,
    MediaRequestError,
    check_pass_budget,
    check_request,
    decode_inputs,
    find_reserved_token,
    item_cost,
    media_summary,
    public_limits,
)
from .genesis_embed_server_runtime import (
    EncodeResult,
    GpuMemoryUnavailable,
    GpuOutOfMemory,
    InputTooLongError,
    TokenCounts,
    count_input_tokens,
    encode_inputs,
    encode_texts,
    get_runtime_status,
    resolve_backend,
)
from .genesis_embed_server_storage import log_embedding

# Routes whose request bodies RequestBodyLimitMiddleware caps at max_request_mb (other routes: a fixed cap).
BODY_LIMITED_PATHS = frozenset({"/embed", "/v1/embeddings"})
BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# After an over-limit body, read on (and discard) up to this many times the limit before answering 413:
# closing while the client still uploads resets the connection and the client never sees the 413.
BODY_DRAIN_FACTOR = 4
BODY_DRAIN_SECONDS = 10.0
# 422 responses list at most this many validation errors.
MAX_VALIDATION_ERRORS = 50

MEDIA_INPUT_KEYS = frozenset({"text", "image", "audio", "video"})
MediaItems = Annotated[List[str], Field(max_length=MAX_MEDIA_ITEMS_PER_FIELD)]


class MediaInput(BaseModel):
    """One input = one embedding: optional text plus images, audio clips and/or one video (as frames)."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, description="Caption/query text; input_type prompts apply to it only.")
    image: Union[str, MediaItems, None] = Field(
        default=None, description="Base64 (or data URI) PNG/JPEG/WEBP/GIF/BMP image, or a list of them."
    )
    audio: Union[str, MediaItems, None] = Field(
        default=None, description="Base64 (or data URI) WAV/FLAC/OGG/MP3 clip, or a list of them."
    )
    video: MediaItems | None = Field(
        default=None, description="One video as a list of pre-sampled base64 PNG/JPEG/WEBP frames of equal size."
    )

    @model_validator(mode="before")
    @classmethod
    def _known_keys_only(cls, data: Any) -> Any:
        # One error for all unknown keys (extra="forbid" alone gives one per key: 100k keys, 100k errors).
        if isinstance(data, dict):
            unknown = [key for key in data if key not in MEDIA_INPUT_KEYS]
            if unknown:
                names = ", ".join(repr(str(key)[:32]) for key in unknown[:3])
                more = f" and {len(unknown) - 3} more" if len(unknown) > 3 else ""
                raise ValueError(f"unknown keys {names}{more}; allowed: text, image, audio, video")
        return data


EmbedInputs = Union[str, MediaInput, Annotated[List[Union[str, MediaInput]], Field(max_length=MAX_INPUTS_PER_REQUEST)]]


class EmbedRequest(BaseModel):
    model: str | None = Field(default=None, description="SentenceTransformer model id")
    inputs: EmbedInputs
    input_type: str | None = Field(
        default=None,
        description=(
            "'query' or 'document' prepends the model's query/document prompt (e.g. 'query: ' for E5). "
            "Aliases: passage, search_query, search_document. Omit or use any other value to embed the raw text."
        ),
    )
    truncate: bool | None = Field(
        default=None,
        description=(
            "true cuts texts at the model's token limit; false rejects them with HTTP 400. "
            "Omit to use the server's too_long_policy setting (default: reject). Never applies to media inputs."
        ),
    )


class EmbedResponse(BaseModel):
    model_id: str
    dimension: int
    vectors: List[List[float]]
    backend: str
    device: str
    device_label: str
    input_count: int
    input_chars: int
    input_tokens: int | None = None
    media: dict[str, Any] | None = None
    load_duration_ms: float
    encode_duration_ms: float
    total_duration_ms: float
    texts_per_second: float | None = None
    batched: bool = False


class OpenAIEmbeddingRequest(BaseModel):
    input: EmbedInputs
    model: str | None = None
    encoding_format: str | None = None
    user: str | None = None
    input_type: str | None = Field(
        default=None,
        description=(
            "'query' or 'document' prepends the model's query/document prompt (e.g. 'query: ' for E5). "
            "Aliases: passage, search_query, search_document. Omit or use any other value to embed the raw text."
        ),
    )
    truncate: bool | None = Field(
        default=None,
        description=(
            "true cuts texts at the model's token limit; false rejects them with HTTP 400. "
            "Omit to use the server's too_long_policy setting (default: reject). Never applies to media inputs."
        ),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Union[str, List[str], None]) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _normalize_inputs(inputs: EmbedInputs) -> list[Union[str, MediaInputData]]:
    normalized = inputs if isinstance(inputs, list) else [inputs]
    return [
        MediaInputData(
            text=item.text,
            image=_as_list(item.image),
            audio=_as_list(item.audio),
            video=list(item.video) if item.video else None,
        )
        if isinstance(item, MediaInput)
        else str(item or "")
        for item in normalized
    ]


# Common client conventions (NVIDIA NIM: passage, Cohere/LiteLLM: search_query/search_document).
_INPUT_TYPE_ALIASES = {
    "query": "query",
    "search_query": "query",
    "document": "document",
    "passage": "document",
    "search_document": "document",
}


def _input_prompt(model_id: str, input_type: str | None) -> str:
    kind = _INPUT_TYPE_ALIASES.get(str(input_type or "").strip().lower())
    if not kind:
        return ""
    return str(get_model_spec(model_id).get(f"{kind}_prompt") or "")


def _prepare_inputs(model_id: str, inputs: list[Any], input_type: str | None) -> list[Any]:
    """Structure and reserved-token checks, prompts on text only. Text-only objects become plain strings."""
    check_reserved = bool(get_model_spec(model_id).get("modalities"))
    prompt = _input_prompt(model_id, input_type)
    prepared: list[Any] = []
    for index, item in enumerate(inputs):
        is_media = isinstance(item, MediaInputData)
        if is_media and item.text is None and not item.has_media():
            raise MediaRequestError(
                "invalid_input", f"Input {index} needs at least one of text, image, audio or video.", index=index
            )
        text = item.text if is_media else item
        token = find_reserved_token(text) if check_reserved else None
        if token:
            raise MediaRequestError(
                "reserved_token", f"Input {index} contains the reserved token {token}.", index=index, token=token
            )
        if is_media and item.has_media():
            # A media-only input gets no prompt.
            prepared.append(dataclasses.replace(item, text=prompt + item.text if item.text else None))
        else:
            prepared.append(prompt + str(text or ""))
    return prepared


def _text_chars(items: list[Any]) -> int:
    return sum(len(item.text or "") if isinstance(item, (MediaInputData, DecodedMedia)) else len(str(item)) for item in items)


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
        "input_tokens": result.input_tokens,
        "media": result.media,
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


class _MediaSlots:
    """Media requests decoding/encoding right now; never waits (the limit is a live setting)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self.active >= limit:
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)


media_slots = _MediaSlots()


def _raise_if_too_long(counted: TokenCounts, check: Callable[[int], bool]) -> None:
    if not counted.max_tokens or not counted.counts:
        return
    too_long = [
        {"index": index, "tokens": tokens}
        for index, tokens in enumerate(counted.counts)
        if tokens > counted.max_tokens and check(index)
    ]
    if too_long:
        raise InputTooLongError(counted.model_id, counted.max_tokens, too_long)


def _with_validation(result: EncodeResult, load_ms: float, input_tokens: int | None) -> EncodeResult:
    # The model may have been loaded while counting tokens; report that time like a load inside encode.
    return dataclasses.replace(
        result,
        load_duration_ms=round(result.load_duration_ms + load_ms, 3),
        total_duration_ms=round(result.total_duration_ms + load_ms, 3),
        input_tokens=input_tokens if input_tokens is not None else result.input_tokens,
    )


async def _embed_texts(
    request: Request, model_id: str, texts: list[str], settings: dict[str, Any], truncate: bool, batched: bool
) -> tuple[EncodeResult, bool]:
    counted: TokenCounts | None = None
    if not truncate:
        # Before batching, so an oversized text never fails other clients' requests in the same batch.
        counted = await asyncio.to_thread(count_input_tokens, model_id, texts, settings)
        _raise_if_too_long(counted, lambda index: True)
    if batched and getattr(request.app.state, "embed_batch_manager", None) is not None:
        result = await request.app.state.embed_batch_manager.enqueue(model_id, texts, settings)
    else:
        result = await asyncio.to_thread(encode_texts, model_id, texts, settings)
        batched = False
    if counted is not None:
        result = _with_validation(result, counted.load_duration_ms, sum(counted.counts) if counted.counts else None)
    return result, batched


async def _embed_media(model_id: str, items: list[Any], settings: dict[str, Any], truncate: bool) -> EncodeResult:
    """Validation pipeline for requests with media; encodes directly (never through the batch manager)."""
    limits = MediaLimits.from_settings(settings)
    spec = get_model_spec(model_id)
    cost_mb = spec.get("media_cost_mb") or {}
    media_items = [(index, item) for index, item in enumerate(items) if isinstance(item, MediaInputData)]
    backend = await asyncio.to_thread(resolve_backend, model_id, settings)
    check_request(
        media_items, model_id=model_id, model_modalities=spec.get("modalities") or [], backend=backend, limits=limits
    )
    for index, item in media_items:
        # Image and frame counts are known before decoding; audio length and tokens are checked after.
        check_pass_budget(index, InputCost(images=len(item.image), video_frames=len(item.video or [])), limits, cost_mb)

    if not media_slots.try_acquire(limits.max_concurrent_media_requests):
        raise MediaRequestError(
            "media_busy",
            "Too many media requests in progress; retry shortly.",
            status_code=429,
            max_concurrent=limits.max_concurrent_media_requests,
        )
    try:
        decoded = await asyncio.to_thread(decode_inputs, items, limits)
        counted = await asyncio.to_thread(count_input_tokens, model_id, decoded, settings)
        is_media = [isinstance(item, DecodedMedia) for item in decoded]
        # Media inputs are never truncated (the processor would fail on cut media tokens).
        _raise_if_too_long(counted, lambda index: is_media[index] or not truncate)
        counts = list(counted.counts)
        if counts and counted.max_tokens:
            counts = [count if is_media[index] else min(count, counted.max_tokens) for index, count in enumerate(counts)]
        for index, item in enumerate(decoded):
            if is_media[index]:
                check_pass_budget(index, item_cost(item, counts[index] if counts else 0), limits, cost_mb)
        result = await asyncio.to_thread(encode_inputs, model_id, decoded, settings, counts or None)
        return _with_validation(result, counted.load_duration_ms, None)
    finally:
        media_slots.release()


def _http_error(exc: Exception, *, has_media: bool) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, MediaRequestError):
        headers = {"Retry-After": "1"} if exc.status_code == 429 else None
        return HTTPException(status_code=exc.status_code, detail=exc.detail(), headers=headers)
    if isinstance(exc, InputTooLongError):
        hint = (
            "Media inputs are never truncated: send fewer images, frames or audio seconds, or shorter text "
            "(\"truncate\": true only cuts plain string inputs)."
            if has_media
            else "Split the text, or send \"truncate\": true to cut it at the limit."
        )
        return HTTPException(
            status_code=400,
            detail={
                "error": "input_too_long",
                "message": f"{exc}. {hint}",
                "model_id": exc.model_id,
                "max_tokens": exc.max_tokens,
                "inputs": exc.too_long,
            },
        )
    if isinstance(exc, GpuOutOfMemory):
        return HTTPException(
            status_code=503,
            detail={"error": "gpu_out_of_memory", "message": f"{exc}; retry later or send smaller requests.", "model_id": exc.model_id},
        )
    if isinstance(exc, GpuMemoryUnavailable):
        return HTTPException(
            status_code=503,
            detail={
                "error": "gpu_memory_unavailable",
                "message": f"Not enough free GPU memory for this request ({exc}); retry later.",
                "required_mb": exc.required_mb,
                "available_mb": exc.available_mb,
            },
        )
    # Media paths: internal exception text could describe payload contents; keep it out of the response.
    return HTTPException(status_code=500, detail="embedding failed" if has_media else f"embedding failed: {exc}")


async def embed_texts_for_request(
    request: Request,
    *,
    model_id: str | None,
    inputs: list[Any],
    route: str,
    force_direct: bool = False,
    input_type: str | None = None,
    truncate: bool | None = None,
) -> tuple[EncodeResult, bool]:
    """Embed strings and/or MediaInputData. Requests with media bypass the batch manager."""
    if not inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")

    with settings_lock:
        settings_snapshot = dict(current_settings)
    effective_model = (model_id or settings_snapshot.get("default_model") or "").strip()
    if not effective_model:
        raise HTTPException(status_code=400, detail="model is required")

    source_ip = request.client.host if request.client else "unknown"
    has_media = any(isinstance(item, MediaInputData) and item.has_media() for item in inputs)
    batched = bool(settings_snapshot.get("batching_enabled")) and not force_direct and not has_media
    if truncate is None:
        truncate = settings_snapshot.get("too_long_policy") == "truncate"
    items = inputs
    try:
        items = _prepare_inputs(effective_model, inputs, input_type)
        if has_media:
            result = await _embed_media(effective_model, items, settings_snapshot, truncate)
        else:
            result, batched = await _embed_texts(request, effective_model, items, settings_snapshot, truncate, batched)
    except Exception as exc:
        entry = {
            "timestamp": _now_iso(),
            "source_ip": source_ip,
            "route": route,
            "model_id": effective_model,
            "input_count": len(inputs),
            "input_chars": _text_chars(items),
            "media": media_summary(inputs),
            "success": False,
            "error": str(exc)[:500],
            "batched": batched,
        }
        _record_history(entry)
        raise _http_error(exc, has_media=has_media) from exc

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
        "input_tokens": result.input_tokens,
        "media": result.media,
        "load_duration_ms": result.load_duration_ms,
        "encode_duration_ms": result.encode_duration_ms,
        "total_duration_ms": result.total_duration_ms,
        "texts_per_second": result.texts_per_second,
        "success": True,
        "batched": batched,
    }
    _record_history(entry)
    return result, batched


def _max_request_bytes() -> int:
    default, minimum, maximum = MEDIA_LIMIT_SETTINGS["max_request_mb"]
    with settings_lock:
        value = current_settings.get("max_request_mb", default)
    try:
        megabytes = int(value)
    except (TypeError, ValueError):
        megabytes = default
    return max(minimum, min(maximum, megabytes)) * 1024 * 1024


class RequestBodyLimitMiddleware:
    """Pure ASGI body cap with HTTP 413: max_request_mb (read live) for the embedding routes, OTHER_ROUTES_MAX_BODY_MB
    for every other POST/PUT/PATCH/DELETE route (FastAPI buffers and parses a body before auth dependencies run).

    Content-Length is checked before reading; chunked bodies are counted while reading, so an oversized upload is
    never buffered beyond the limit. The rest of an oversized body is read and dropped (bounded) before the 413:
    closing while the client still uploads resets the connection and the client never sees the response.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") not in BODY_METHODS:
            await self.app(scope, receive, send)
            return
        setting = "max_request_mb" if scope.get("path") in BODY_LIMITED_PATHS else None
        max_bytes = _max_request_bytes() if setting else OTHER_ROUTES_MAX_BODY_MB * 1024 * 1024
        drain_bytes = max_bytes * BODY_DRAIN_FACTOR
        declared: int | None = None
        expects_continue = False
        for name, value in scope.get("headers") or []:
            name = name.lower()
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
            elif name == b"expect":
                expects_continue = value.lower() == b"100-continue"
        if declared is not None and declared > max_bytes:
            # A client waiting for 100-continue has sent nothing yet; a body far over the limit is not worth reading.
            if not expects_continue and declared <= drain_bytes:
                await self._drain(receive, drain_bytes)
            await self._reject(send, max_bytes, setting)
            return

        chunks: list[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            more_body = bool(message.get("more_body", False))
            received += len(chunk)
            if received > max_bytes:
                chunks.clear()
                if more_body:
                    await self._drain(receive, drain_bytes - received)
                await self._reject(send, max_bytes, setting)
                return
            chunks.append(chunk)
            if not more_body:
                break
        body = b"".join(chunks)
        chunks.clear()
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _drain(receive: Any, max_bytes: int) -> None:
        """Read and drop the rest of the body: up to max_bytes, BODY_DRAIN_SECONDS or the last chunk."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + BODY_DRAIN_SECONDS
        drained = 0
        while drained <= max_bytes:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                return
            if message["type"] != "http.request":
                return
            drained += len(message.get("body", b""))
            if not message.get("more_body", False):
                return

    @staticmethod
    async def _reject(send: Any, max_bytes: int, setting: str | None) -> None:
        source = f" ({setting})" if setting else ""
        payload = json.dumps(
            {
                "detail": {
                    "error": "request_too_large",
                    "message": f"Request body exceeds {max_bytes // (1024 * 1024)} MB{source}.",
                    "max_bytes": max_bytes,
                }
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Like FastAPI's default, minus "input"/"ctx": they would echo megabytes of base64 back to the client.
    # Capped: the schema caps list lengths, but a 422 never needs more than the first errors.
    errors = [
        {key: error[key] for key in ("loc", "msg", "type") if key in error} for error in exc.errors()[:MAX_VALIDATION_ERRORS]
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


def _media_health(settings_snapshot: dict[str, Any], backend: str | None) -> dict[str, Any]:
    limits = MediaLimits.from_settings(settings_snapshot)
    spec = get_model_spec(str(settings_snapshot.get("default_model") or ""))
    modalities = [kind for kind in spec.get("modalities") or [] if limits.enabled(kind)] if backend == "pytorch" else []
    return {"modalities": modalities, "limits": public_limits(limits)}


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
            "media": _media_health(settings_snapshot, (runtime.get("resolved_runtime") or {}).get("backend")),
            **runtime,
        }

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: Request, payload: EmbedRequest):
        api_key_id = authorize_api_key(request)
        result, batched = await embed_texts_for_request(
            request,
            model_id=payload.model,
            inputs=_normalize_inputs(payload.inputs),
            route="/embed",
            input_type=payload.input_type,
            truncate=payload.truncate,
        )
        if api_key_id:
            get_auth_store().record_api_key_usage(api_key_id, result.input_count)
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
        api_key_id = authorize_api_key(request)
        result, _ = await embed_texts_for_request(
            request,
            model_id=payload.model,
            inputs=_normalize_inputs(payload.input),
            route="/v1/embeddings",
            input_type=payload.input_type,
            truncate=payload.truncate,
        )
        if api_key_id:
            get_auth_store().record_api_key_usage(api_key_id, result.input_count)
        # Counted tokens when known (validated or media requests), else the old chars/4 approximation.
        prompt_tokens = result.input_tokens if result.input_tokens is not None else max(1, result.input_chars // 4)
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
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    return app
