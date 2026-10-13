"""Over-limit texts for a model that is not loaded are rejected by its tokenizer alone, without loading it."""
from __future__ import annotations

import pytest

from backend import genesis_embed_server_runtime as runtime
from backend.genesis_embed_server_globals import get_model_spec

E5_LARGE = "intfloat/multilingual-e5-large"
GEMMA = "google/embeddinggemma-2"
CUSTOM = "someone/custom-model"
LONG = "word " * 600  # 602 fake tokens with <bos>/<eos>


def post(client, inputs, model=E5_LARGE, route="/embed", **extra):
    key = "input" if route == "/v1/embeddings" else "inputs"
    return client.post(route, json={"model": model, key: inputs, **extra})


def expected_detail(model_id, max_tokens, too_long):
    return {
        "error": "input_too_long",
        "message": f"{len(too_long)} input(s) exceed the token limit of {max_tokens} for '{model_id}'. "
        "Split the text, or send \"truncate\": true to cut it at the limit.",
        "model_id": model_id,
        "max_tokens": max_tokens,
        "inputs": too_long,
    }


@pytest.mark.parametrize("route", ["/embed", "/v1/embeddings"])
def test_over_limit_request_does_not_load_the_model(client, fake_model, route):
    response = post(client, ["short", LONG], route=route)
    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail(E5_LARGE, 512, [{"index": 1, "tokens": 602}])
    assert runtime._model_cache == {} and fake_model.calls == []
    assert fake_model.tokenizer_loads == [E5_LARGE]


def test_rejection_matches_the_loaded_model(client, fake_model):
    unloaded = post(client, [LONG, "x", LONG])
    assert post(client, ["fits"]).status_code == 200  # loads the model
    (entry,) = runtime._model_cache.values()
    entry["native_max_seq_length"] = 512  # the real e5 window (the fake model reports 8192)
    loaded = post(client, [LONG, "x", LONG])
    assert unloaded.status_code == loaded.status_code == 400
    assert unloaded.json() == loaded.json()
    # Loaded models count with their own tokenizer; the standalone one was loaded only once.
    assert fake_model.tokenizer_loads == [E5_LARGE]


def test_tokenizer_rejection_honors_window_setting_and_prompt(client, fake_model, settings):
    settings(max_seq_length=100)
    response = post(client, ["word " * 99], input_type="query")  # "query: " prompt adds a token
    assert response.status_code == 400
    assert response.json()["detail"]["inputs"] == [{"index": 0, "tokens": 102}] and runtime._model_cache == {}


def test_texts_that_fit_load_and_encode_as_before(client, fake_model):
    response = post(client, ["word " * 500])
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] == 502 and len(runtime._model_cache) == 1


def test_truncate_skips_the_check(client, fake_model):
    assert post(client, [LONG], truncate=True).status_code == 200
    assert fake_model.tokenizer_loads == []


def test_unknown_window_falls_back_to_loading(client, fake_model, settings):
    assert not get_model_spec(CUSTOM).get("max_seq_length")
    settings(max_seq_length=100)
    response = post(client, [LONG], model=CUSTOM)
    assert response.status_code == 400 and response.json()["detail"]["max_tokens"] == 100
    assert len(runtime._model_cache) == 1 and fake_model.tokenizer_loads == []


def test_unavailable_tokenizer_falls_back_to_loading(client, fake_model, monkeypatch):
    def fail(model_id, cache_folder):
        raise OSError("offline")

    monkeypatch.setattr(runtime, "_load_standalone_tokenizer", fail)
    # The fake model's window is 8192; loading it is the old behavior.
    assert post(client, [LONG]).status_code == 200 and len(runtime._model_cache) == 1


def test_loaded_model_skips_the_standalone_tokenizer(client, fake_model, settings):
    settings(max_seq_length=100)
    assert post(client, ["x"], model=GEMMA).status_code == 200
    fake_model.tokenizer_loads.clear()
    response = post(client, [LONG], model=GEMMA)
    assert response.status_code == 400 and fake_model.tokenizer_loads == []


def test_standalone_tokenizer_cache_is_bounded(fake_model):
    models = [f"org/model-{index}" for index in range(runtime._STANDALONE_TOKENIZERS_MAX + 2)]
    with runtime._standalone_tokenizer_lock:
        for model_id in models + [models[-1]]:
            runtime._standalone_tokenizer(model_id, {})
    assert len(runtime._standalone_tokenizers) == runtime._STANDALONE_TOKENIZERS_MAX
    assert fake_model.tokenizer_loads == models


def test_validate_input_lengths_rejects_without_loading(fake_model):
    with pytest.raises(runtime.InputTooLongError) as error:
        runtime.validate_input_lengths(E5_LARGE, ["x", LONG])
    assert error.value.too_long == [{"index": 1, "tokens": 602}] and error.value.max_tokens == 512
    assert runtime._model_cache == {}


def test_models_endpoint_reports_e5_window_before_loading(client):
    data = {entry["id"]: entry for entry in client.get("/v1/models").json()["data"]}
    assert data[E5_LARGE]["max_model_len"] == 512
    assert runtime._model_cache == {}


def test_media_requests_never_load_a_text_only_model(client, fake_model):
    from media_samples import png_b64

    response = post(client, [{"text": LONG, "image": png_b64()}])
    assert response.status_code == 400 and response.json()["detail"]["error"] == "media_unsupported"
    assert runtime._model_cache == {}

