"""Public API with media inputs: validation pipeline, error mapping, routing, usage and history.

Runs the real api/runtime/media code against a FakeModel (conftest), so no model is loaded.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from backend import genesis_embed_server_api as api
from backend import genesis_embed_server_runtime as runtime
from backend.genesis_embed_server_globals import embedding_history
from backend.genesis_embed_server_media import audio_tokens, image_tokens, video_tokens
from conftest import FAKE_DIMENSION
from media_samples import b64, frames_b64, image_bytes, png_b64, wav_b64

GEMMA = "google/embeddinggemma-2"
E5 = "intfloat/multilingual-e5-small"


def post(client, inputs, route="/embed", **extra):
    key = "input" if route == "/v1/embeddings" else "inputs"
    return client.post(route, json={"model": extra.pop("model", GEMMA), key: inputs, **extra})


def detail(response):
    return response.json()["detail"]


class RecordingBatchManager:
    def __init__(self):
        self.calls = []

    async def enqueue(self, model_id, texts, settings):
        self.calls.append(list(texts))
        return runtime.encode_texts(model_id, texts, settings)


# --- text inputs are never handed to sentence-transformers as bare strings --------------------------------

@pytest.mark.parametrize("route", ["/embed", "/v1/embeddings"])
def test_url_and_path_texts_reach_model_wrapped(client, fake_model, route):
    texts = ["https://x/y.jpg", "C:\\x.png", "data:image/png;base64,iVBORw0KGgo="]
    response = post(client, texts, route=route)
    assert response.status_code == 200, response.text
    assert fake_model.calls[-1]["inputs"] == [{"text": text} for text in texts]


def test_url_text_wrapped_through_batch_manager(client, fake_model, settings):
    settings(batching_enabled=True)
    manager = RecordingBatchManager()
    client.app.state.embed_batch_manager = manager
    response = post(client, ["https://169.254.169.254/x.jpg"])
    assert response.status_code == 200 and response.json()["batched"] is True
    assert manager.calls == [["https://169.254.169.254/x.jpg"]]
    assert fake_model.calls[-1]["inputs"] == [{"text": "https://169.254.169.254/x.jpg"}]


def test_text_only_object_is_plain_text(client, fake_model):
    response = post(client, [{"text": "hello there"}], input_type="query")
    assert response.status_code == 200
    assert fake_model.calls[-1]["inputs"] == [{"text": "task: search result | query: hello there"}]
    assert response.json()["media"] is None


# --- mixed requests ------------------------------------------------------------------------------------

def test_mixed_request_order_and_batch_bypass(client, fake_model, settings):
    settings(batching_enabled=True)
    manager = RecordingBatchManager()
    client.app.state.embed_batch_manager = manager
    inputs = [
        "first text",
        {"image": [png_b64(), png_b64()]},
        "second",
        {"text": "cap", "audio": wav_b64(1.0)},
        {"video": frames_b64(3)},
    ]
    response = post(client, inputs)
    assert response.status_code == 200, response.text
    body = response.json()
    assert manager.calls == []  # media requests never enter the batch manager
    assert body["batched"] is False and body["input_count"] == 5 and body["dimension"] == FAKE_DIMENSION
    tags = [vector[0] for vector in body["vectors"]]
    assert tags == [len("first text"), 2000, len("second"), 100 + 3, 30]
    media_items = [item for call in fake_model.media_calls() for item in call["inputs"]]
    assert [list(item) for item in media_items] == [["image"], ["text", "audio"], ["video"]]
    assert all(call["batch_size"] == len(call["inputs"]) for call in fake_model.media_calls())
    assert body["media"] == {"images": 2, "audio_seconds": 1.0, "video_frames": 3}
    assert body["input_chars"] == len("first text") + len("second") + len("cap")


def test_prompt_applies_to_text_part_only(client, fake_model):
    response = post(client, [{"text": "red shoe", "image": png_b64()}, {"image": png_b64()}], input_type="document")
    assert response.status_code == 200
    first, second = fake_model.media_calls()[0]["inputs"]
    assert first["text"] == "title: none | text: red shoe" and list(first) == ["text", "image"]
    assert list(second) == ["image"]


def test_media_chunks_respect_budget(client, fake_model, settings):
    settings(media_pass_budget_mb=300, max_images_per_request=32)
    # 3 images per input: 3 x 74 + 50 = 272 <= 300, two inputs would be 494 -> one input per pass.
    response = post(client, [{"image": [png_b64()] * 3} for _ in range(4)])
    assert response.status_code == 200, response.text
    assert [call["batch_size"] for call in fake_model.media_calls()] == [1, 1, 1, 1]


# --- usage, response fields, history ----------------------------------------------------------------------

def test_usage_tokens_for_media(client, fake_model):
    response = post(client, [{"text": "two words", "image": b64(image_bytes(800, 600))}, "three more words"], route="/v1/embeddings")
    assert response.status_code == 200, response.text
    expected = (2 + 2 + image_tokens(600, 800)) + (3 + 2)
    assert response.json()["usage"] == {"prompt_tokens": expected, "total_tokens": expected}


def test_usage_tokens_for_validated_text(client, fake_model):
    response = post(client, ["one two three", "four"], route="/v1/embeddings")
    assert response.json()["usage"]["prompt_tokens"] == (3 + 2) + (1 + 2)


def test_usage_falls_back_to_chars_when_not_counted(client, fake_model):
    response = post(client, ["x" * 40], route="/v1/embeddings", truncate=True)
    assert response.json()["usage"]["prompt_tokens"] == 10


def test_response_reports_tokens_and_media(client, fake_model):
    audio = wav_b64(2.0)
    frames = frames_b64(4, 640, 480)
    response = post(client, [{"audio": audio, "video": frames}])
    body = response.json()
    assert body["input_tokens"] == 2 + audio_tokens(32000) + video_tokens(4, 480, 640)
    assert body["media"] == {"images": 0, "audio_seconds": 2.0, "video_frames": 4}


def test_api_key_usage_counts_inputs(client, fake_model, monkeypatch):
    recorded = []

    class Store:
        def record_api_key_usage(self, key_id, items):
            recorded.append((key_id, items))

    monkeypatch.setattr(api, "authorize_api_key", lambda request: "key-1")
    monkeypatch.setattr(api, "get_auth_store", lambda: Store())
    assert post(client, ["a", {"image": png_b64()}]).status_code == 200
    assert recorded == [("key-1", 2)]


def test_history_has_counts_but_no_payload(client, fake_model, logged):
    image = png_b64(90, 90)
    audio = wav_b64(1.0)
    assert post(client, [{"text": "c", "image": image, "audio": audio}]).status_code == 200
    assert post(client, [{"image": "!!" + image}]).status_code == 400
    dumped = json.dumps(list(embedding_history)) + json.dumps(logged)
    assert image[:64] not in dumped and audio[:64] not in dumped
    success = [entry for entry in logged if entry["success"]][0]
    assert success["media"] == {"images": 1, "audio_seconds": 1.0, "video_frames": 0}
    assert success["input_tokens"] > 0
    failure = [entry for entry in logged if not entry["success"]][0]
    assert failure["media"] == {"images": 1, "audio_seconds": None, "video_frames": 0}


# --- rejected requests: the encoder is never called ---------------------------------------------------------

def tiff_b64():
    return b64(image_bytes(16, 16, fmt="TIFF"))


# Limits that can only be checked on decoded data (pixel count from the header, audio durations).
DECODED_LIMITS = {"max_image_megapixels", "max_audio_seconds_per_clip", "max_audio_seconds_per_request"}


def decoded_before_limit_check(*args, **kwargs):
    raise AssertionError("decoded before the limit check")


@pytest.mark.parametrize(
    "inputs,overrides,model,status,error,extra",
    [
        ([{}], {}, GEMMA, 400, "invalid_input", {"index": 0}),
        (["ok", {"image": []}], {}, GEMMA, 400, "invalid_input", {"index": 1}),
        ([{"image": "%%%not-base64"}], {}, GEMMA, 400, "invalid_media", {"index": 0, "field": "image", "position": 0}),
        ([{"image": [png_b64(), tiff_b64()]}], {}, GEMMA, 400, "invalid_media", {"field": "image", "position": 1}),
        ([{"audio": wav_b64(0.5, 44101)}], {}, GEMMA, 400, "invalid_media", {"field": "audio"}),
        ([{"video": [png_b64(64, 48), png_b64(48, 64)]}], {}, GEMMA, 400, "invalid_media", {"field": "video", "position": 1}),
        ([{"image": png_b64()}], {"image_inputs_enabled": False}, GEMMA, 400, "media_disabled", {"modality": "image"}),
        ([{"video": frames_b64(2)}], {"video_inputs_enabled": False}, GEMMA, 400, "media_disabled", {"modality": "video"}),
        ([{"image": png_b64()}], {}, E5, 400, "media_unsupported", {"model_id": E5, "modality": "image"}),
        ([{"image": [png_b64()] * 5}], {}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_images_per_input", "value": 5, "max": 4, "index": 0}),
        ([{"image": [png_b64()] * 4}] * 5, {}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_images_per_request", "value": 20, "max": 16}),
        ([{"audio": [wav_b64(0.5)] * 3}], {}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_audio_clips_per_input"}),
        ([{"video": frames_b64(9)}], {}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_video_frames", "value": 9, "max": 8}),
        ([{"video": frames_b64(2)}, {"video": frames_b64(2)}], {}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_videos_per_request"}),
        ([{"image": b64(b"\x00" * (1024 * 1024 + 4096))}], {"max_media_item_mb": 1}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_media_item_mb"}),
        ([{"image": b64(image_bytes(1500, 1000))}], {"max_image_megapixels": 1}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_image_megapixels"}),
        ([{"audio": wav_b64(3.0)}], {"max_audio_seconds_per_clip": 2}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_audio_seconds_per_clip"}),
        ([{"audio": [wav_b64(2.0), wav_b64(2.0)]}], {"max_audio_seconds_per_request": 3}, GEMMA, 400, "media_limit_exceeded", {"limit": "max_audio_seconds_per_request"}),
        ([{"image": [png_b64()] * 4}], {"media_pass_budget_mb": 200}, GEMMA, 400, "media_limit_exceeded", {"limit": "media_pass_budget_mb", "max": 200, "index": 0}),
        (["hello <|image|> world"], {}, GEMMA, 400, "reserved_token", {"index": 0, "token": "<|image|>"}),
        (["fine", {"text": "x <|video|>", "image": png_b64()}], {}, GEMMA, 400, "reserved_token", {"index": 1, "token": "<|video|>"}),
    ],
)
def test_rejections_never_call_encoder(client, fake_model, settings, monkeypatch, inputs, overrides, model, status, error, extra):
    settings(**overrides)
    # Structure, support, count, size and pre-decode budget checks must fail before anything is decoded.
    if not (error == "invalid_media" or extra.get("limit") in DECODED_LIMITS):
        monkeypatch.setattr(api, "decode_inputs", decoded_before_limit_check)
    response = post(client, inputs, model=model)
    assert response.status_code == status, response.text
    body = detail(response)
    assert body["error"] == error and body["message"]
    for key, value in extra.items():
        assert body[key] == value, (key, body)
    assert fake_model.calls == []


def test_media_unsupported_on_openvino(client, fake_model, monkeypatch):
    monkeypatch.setattr(api, "resolve_backend", lambda model_id, settings=None: "openvino")
    response = post(client, [{"audio": wav_b64(0.5)}])
    assert response.status_code == 400 and detail(response)["error"] == "media_unsupported"
    assert fake_model.calls == []


def test_reserved_token_not_checked_for_text_only_models(client, fake_model):
    assert post(client, ["hello <|image|>"], model=E5).status_code == 200


def test_truncate_does_not_bypass_media_token_limit(client, fake_model, settings):
    settings(max_seq_length=300)
    response = post(client, [{"text": "x", "image": [png_b64(), png_b64()]}], truncate=True)
    assert response.status_code == 400
    body = detail(response)
    assert body["error"] == "input_too_long" and body["max_tokens"] == 300
    assert body["inputs"] == [{"index": 0, "tokens": 2 + 1 + 2 * image_tokens(48, 64)}]
    assert fake_model.calls == []


def test_truncate_still_applies_to_plain_strings_in_media_requests(client, fake_model, settings):
    settings(max_seq_length=300)
    long_text = "word " * 400
    response = post(client, [long_text, {"image": png_b64()}], truncate=True)
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] == 300 + 2 + image_tokens(48, 64)
    response = post(client, [long_text, {"image": png_b64()}], truncate=False)
    assert response.status_code == 400 and detail(response)["inputs"] == [{"index": 0, "tokens": 402}]


def test_media_busy_429(client, fake_model, settings, monkeypatch):
    settings(max_concurrent_media_requests=1)
    assert api.media_slots.try_acquire(1)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(api, "decode_inputs", decoded_before_limit_check)
            response = post(client, [{"image": png_b64()}])
    finally:
        api.media_slots.release()
    assert response.status_code == 429 and response.headers["retry-after"] == "1"
    assert detail(response)["error"] == "media_busy"
    assert fake_model.calls == []
    # Text-only requests never take a media slot.
    assert api.media_slots.try_acquire(1)
    try:
        assert post(client, ["text"]).status_code == 200
    finally:
        api.media_slots.release()


def test_media_slot_released_after_errors(client, fake_model, settings):
    settings(max_concurrent_media_requests=1)
    assert post(client, [{"image": "%%%"}]).status_code == 400
    assert post(client, [{"image": png_b64()}]).status_code == 200
    assert api.media_slots.active == 0


def test_body_too_large_by_content_length(client, fake_model, settings):
    settings(max_request_mb=1)
    response = post(client, [{"image": b64(b"\x01" * (1024 * 1024))}])
    assert response.status_code == 413
    assert detail(response) == {
        "error": "request_too_large",
        "message": detail(response)["message"],
        "max_bytes": 1024 * 1024,
    }
    assert fake_model.calls == []


def test_body_too_large_chunked(client, fake_model, settings):
    settings(max_request_mb=1)
    payload = json.dumps({"model": GEMMA, "inputs": ["x" * (1024 * 1024)]}).encode()

    def chunks():
        for start in range(0, len(payload), 64 * 1024):
            yield payload[start : start + 64 * 1024]

    response = client.post("/v1/embeddings", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413 and detail(response)["error"] == "request_too_large"
    assert fake_model.calls == []


def test_body_limit_allows_normal_and_chunked_small_bodies(client, fake_model):
    payload = json.dumps({"model": GEMMA, "inputs": ["small"]}).encode()
    response = client.post("/embed", content=iter([payload[:10], payload[10:]]), headers={"content-type": "application/json"})
    assert response.status_code == 200, response.text


def test_invalid_media_reason_is_the_bare_cause(client, fake_model):
    for field, value in (("image", "data:image/png;base64,"), ("video", ["data:image/png;base64,"])):
        response = post(client, [{field: value}])
        assert response.status_code == 400
        assert detail(response)["message"] == f"Input 0 {field}[0]: empty payload."
        assert detail(response)["reason"] == "empty payload"


def test_validation_error_does_not_echo_input(client, fake_model):
    secret = b64(b"S" * 3000)
    response = client.post("/embed", json={"model": GEMMA, "inputs": [{"image": secret, "bogus": secret}]})
    assert response.status_code == 422
    assert secret[:40] not in response.text
    for error in detail(response):
        assert set(error) <= {"loc", "msg", "type"}
    assert fake_model.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "torch.OutOfMemoryError",
        # Allocation failures outside the caching allocator are plain RuntimeErrors.
        "CUDA error: out of memory",
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_gpu_out_of_memory_maps_to_503(client, fake_model, monkeypatch, message):
    import torch

    released = []
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: released.append(True))
    if message == "torch.OutOfMemoryError":
        fake_model.fail_with = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
    else:
        fake_model.fail_with = RuntimeError(message)
    for inputs in ([{"image": png_b64()}], ["just text"]):
        response = post(client, inputs)
        assert response.status_code == 503, response.text
        assert detail(response)["error"] == "gpu_out_of_memory" and detail(response)["model_id"] == GEMMA
    assert len(released) >= 2


def test_gpu_memory_precheck_maps_to_503(client, fake_model, monkeypatch):
    import torch

    monkeypatch.setattr(runtime, "_is_cuda_pytorch", lambda choice: True)
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (300 * 1024 * 1024, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 100 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 50 * 1024 * 1024)
    response = post(client, [{"image": [png_b64()] * 4}])
    assert response.status_code == 503, response.text
    body = detail(response)
    # 4 x 74 + 50 = 346 MB estimate + 256 MB margin; 300 + 50 MB available.
    assert body["error"] == "gpu_memory_unavailable"
    assert body["required_mb"] == 602.0 and body["available_mb"] == 350.0
    assert fake_model.media_calls() == []


def test_media_500_hides_exception_text(client, fake_model):
    fake_model.fail_with = RuntimeError("internal detail QUJDREVG")
    response = post(client, [{"image": png_b64()}])
    assert response.status_code == 500 and detail(response) == "embedding failed"


# --- health -------------------------------------------------------------------------------------------------

def fake_runtime_status(backend="pytorch"):
    return lambda: {"loaded_models": [], "last_runtime_error": None, "resolved_runtime": {"backend": backend}, "system": {}}


def test_health_media_block(client, monkeypatch, settings):
    monkeypatch.setattr(api, "get_runtime_status", fake_runtime_status())
    media = client.get("/health").json()["media"]
    assert media["modalities"] == ["image", "audio", "video"]
    assert media["limits"]["max_video_frames"] == 8 and media["limits"]["max_request_mb"] == 32
    assert media["limits"]["audio_sample_rates"][0] == 8000
    settings(video_inputs_enabled=False, max_video_frames=3)
    media = client.get("/health").json()["media"]
    assert media["modalities"] == ["image", "audio"] and media["limits"]["max_video_frames"] == 3
    monkeypatch.setattr(api, "get_runtime_status", fake_runtime_status("openvino"))
    assert client.get("/health").json()["media"]["modalities"] == []
    settings(default_model=E5)
    monkeypatch.setattr(api, "get_runtime_status", fake_runtime_status())
    assert client.get("/health").json()["media"]["modalities"] == []


# --- 422 fan-out ----------------------------------------------------------------------------------------------

def test_422_for_huge_bad_list_is_small_and_fast(client, fake_model):
    started = time.perf_counter()
    response = client.post("/embed", json={"model": GEMMA, "inputs": [0] * 200_000})
    assert response.status_code == 422
    assert time.perf_counter() - started < 5
    assert len(detail(response)) <= api.MAX_VALIDATION_ERRORS and len(response.content) < 20_000
    assert any(error["type"] == "too_long" for error in detail(response))


@pytest.mark.parametrize("route", ["/embed", "/v1/embeddings"])
def test_422_for_many_unknown_keys_is_one_error(client, fake_model, route):
    response = post(client, {f"key{index}": 0 for index in range(20_000)}, route=route)
    assert response.status_code == 422
    errors = [error for error in detail(response) if "MediaInput" in error["loc"]]
    assert len(errors) == 1 and "and 19997 more" in errors[0]["msg"]
    assert len(response.content) < 5_000


def test_inputs_and_media_lists_are_capped(client, fake_model):
    from backend.genesis_embed_server_globals import MAX_INPUTS_PER_REQUEST, MAX_MEDIA_ITEMS_PER_FIELD

    assert post(client, ["x"] * (MAX_INPUTS_PER_REQUEST + 1)).status_code == 422
    assert post(client, [{"image": ["AAAA"] * (MAX_MEDIA_ITEMS_PER_FIELD + 1)}]).status_code == 422
    assert post(client, [{"video": ["AAAA"] * (MAX_MEDIA_ITEMS_PER_FIELD + 1)}]).status_code == 422
    assert fake_model.calls == []
    assert post(client, ["x"] * MAX_INPUTS_PER_REQUEST).status_code == 200


def test_validation_handler_caps_errors():
    from fastapi.exceptions import RequestValidationError

    errors = [{"loc": ("body", index), "msg": "bad", "type": "x", "input": "secret"} for index in range(1000)]
    response = asyncio.run(api.request_validation_exception_handler(None, RequestValidationError(errors)))
    payload = json.loads(response.body)["detail"]
    assert len(payload) == api.MAX_VALIDATION_ERRORS and "input" not in payload[0]


# --- RequestBodyLimitMiddleware as a plain ASGI app (TestClient always sends the body as one message) -----------

def run_middleware(messages, path="/embed", method="POST", headers=()):
    """Feed body messages to the middleware; returns (bodies the inner app got, sent messages, unread messages)."""
    queue = list(messages)
    inner_calls = []
    sent = []

    async def inner(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        inner_calls.append(body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        if not queue:
            raise AssertionError("read past the end of the body")
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": list(headers)}
    asyncio.run(api.RequestBodyLimitMiddleware(inner)(scope, receive, send))
    return inner_calls, sent, queue


def body_messages(chunks):
    return [{"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1} for index, chunk in enumerate(chunks)]


def test_middleware_counts_bytes_across_chunks(settings):
    settings(max_request_mb=1)
    inner_calls, sent, unread = run_middleware(body_messages([b"x" * (300 * 1024)] * 6))  # 1.8 MB in 300 KB pieces
    assert inner_calls == [] and sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"]["max_bytes"] == 1024 * 1024
    # The rest of the body was read (and dropped) before the 413, so the client can finish its upload and read it.
    assert unread == []


def test_middleware_replays_multi_chunk_body_unchanged(settings):
    settings(max_request_mb=1)
    chunks = [bytes([index]) * (200 * 1024) for index in range(5)]  # 1000 KB, just under 1 MiB
    inner_calls, sent, unread = run_middleware(body_messages(chunks))
    assert inner_calls == [b"".join(chunks)] and sent[0]["status"] == 200 and unread == []


def test_middleware_drain_is_bounded(settings):
    settings(max_request_mb=1)
    inner_calls, sent, unread = run_middleware(body_messages([b"x" * (1024 * 1024)] * 20))
    assert inner_calls == [] and sent[0]["status"] == 413
    assert 0 < len(unread) <= 20 - api.BODY_DRAIN_FACTOR  # stops reading after about 4x the limit


def test_middleware_content_length_drains_before_413_unless_100_continue(settings):
    settings(max_request_mb=1)
    length = (b"content-length", str(2 * 1024 * 1024).encode())
    inner_calls, sent, unread = run_middleware(body_messages([b"x" * (1024 * 1024)] * 2), headers=[length])
    assert inner_calls == [] and sent[0]["status"] == 413 and unread == []
    # A client waiting for "100 Continue" has not sent the body: answer right away without reading.
    headers = [length, (b"expect", b"100-continue")]
    inner_calls, sent, unread = run_middleware(body_messages([b"x" * (1024 * 1024)] * 2), headers=headers)
    assert inner_calls == [] and sent[0]["status"] == 413 and len(unread) == 2


@pytest.mark.parametrize("path,method", [("/api/admin/auth/login", "POST"), ("/api/admin/settings", "PUT"), ("/api/admin/benchmark", "POST")])
def test_middleware_caps_other_routes(path, method):
    from backend.genesis_embed_server_globals import OTHER_ROUTES_MAX_BODY_MB

    limit = OTHER_ROUTES_MAX_BODY_MB * 1024 * 1024
    inner_calls, sent, _ = run_middleware(body_messages([b"x" * limit, b"y"]), path=path, method=method)
    assert inner_calls == [] and sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"] == {
        "error": "request_too_large",
        "message": f"Request body exceeds {OTHER_ROUTES_MAX_BODY_MB} MB.",
        "max_bytes": limit,
    }
    inner_calls, sent, _ = run_middleware(body_messages([b"{}"]), path=path, method=method)
    assert inner_calls == [b"{}"] and sent[0]["status"] == 200


def test_middleware_ignores_get():
    _, sent, _ = run_middleware(body_messages([b"x" * (8 * 1024 * 1024)]), path="/health", method="GET")
    assert sent[0]["status"] == 200
