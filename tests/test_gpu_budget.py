"""Opt-in GPU budget test: near-budget ALLOWED media requests stay within media_pass_budget_mb.

Loads google/embeddinggemma-2 (bf16, eager) on CUDA, capped to ~4 GiB with set_per_process_memory_fraction,
and sends requests through the real API, validation, chunking and encode path. Only runs with
G3_EMBED_GPU_TESTS=1 and at least 3 GiB free VRAM (in cmd.exe quote the set, else the value keeps a trailing space):

    set "G3_EMBED_GPU_TESTS=1" && venv\\Scripts\\python.exe -m pytest tests/test_gpu_budget.py -q -s

Every case plans a pass of at least NEAR_BUDGET x the budget. test_cases_plan_near_budget checks that on CPU
(tokenizer only, skipped without the model files), so an edit to CASES or the limits cannot quietly turn the GPU
test into a low-load check.
"""
from __future__ import annotations

import os

import pytest

from backend import genesis_embed_server_api as api
from backend import genesis_embed_server_runtime as runtime
from backend.genesis_embed_server_globals import MEDIA_LIMIT_SETTINGS, MODEL_SPECS, current_settings
from backend.genesis_embed_server_media import (
    MediaInputData,
    MediaLimits,
    decode_inputs,
    estimate_pass_mb,
    input_tokens,
    item_cost,
    plan_chunks,
)
from backend.genesis_embed_server_storage import normalize_settings
from conftest import PROJECT_ROOT, gpu_tests_enabled, model_files_cached
from media_samples import frames_b64, png_b64, wav_b64

gpu_only = pytest.mark.skipif(not gpu_tests_enabled(), reason="set G3_EMBED_GPU_TESTS=1 to run the GPU budget test")

MODEL = "google/embeddinggemma-2"
COST_MB = MODEL_SPECS[MODEL]["media_cost_mb"]
MIB = 1024 * 1024
GIB = 1024 * MIB
PROCESS_CAP_BYTES = 4 * GIB
RELEASE_SLACK_MB = 64
# Each case must plan at least one pass this close to the budget.
NEAR_BUDGET = 0.85

# Every limit at its hard maximum; only media_pass_budget_mb keeps its default.
GPU_SETTINGS = {
    "default_model": MODEL,
    "execution_target": "nvidia",
    "backend_override": "pytorch",
    "precision": "bfloat16",
    "compile_mode": "off",
    "batching_enabled": False,
    "model_cache_path": str(PROJECT_ROOT / "models"),
    **{name: maximum for name, (_, _, maximum) in MEDIA_LIMIT_SETTINGS.items()},
    "media_pass_budget_mb": MEDIA_LIMIT_SETTINGS["media_pass_budget_mb"][0],
}
BUDGET_MB = GPU_SETTINGS["media_pass_budget_mb"]


def long_text(words: int) -> str:
    return "hello " * words


def _audio_300s_4_4_2():
    clip = wav_b64(30.0, 16000)
    return [{"audio": [clip] * 4}, {"audio": [clip] * 4}, {"audio": [clip] * 2}]


def _audio_mixed_lengths():
    short = wav_b64(0.1, 16000)
    return [{"audio": wav_b64(30.0, 16000)}] + [{"audio": [short] * 3} for _ in range(11)]


# Built lazily: the CPU suite collects this module too.
CASES = {
    # 32 single-image inputs (request cap): image-tower passes of 15 images (1160 MB).
    "images_32x1": lambda: [{"text": "photo", "image": png_b64(800, 600, (i * 7 % 256, 90, 40))} for i in range(32)],
    # 300 s (request cap) as 4 + 4 + 2 clips of 30 s: one pass, LM term dominant (3 rows x 3010 tokens, 1179 MB).
    "audio_300s_4_4_2": _audio_300s_4_4_2,
    # One 30 s clip, then 11 inputs of 3 x 0.1 s: every clip of a pass is padded to 30 s (10 clips, 1100 MB).
    "audio_mixed_len_30s_and_0.1s": _audio_mixed_lengths,
    # 16 frames at 4:3 plus 7 images in one input, near 4096 tokens (1144 MB).
    "video_16f_plus_7img": lambda: [{"image": [png_b64(640, 480)] * 7, "video": frames_b64(16, 640, 480)}],
    # Two rows of text plus an image near 4096 tokens: multi-row LM pass (about 1070 MB).
    "text_image_near_4096_x2": lambda: [{"text": long_text(3800), "image": png_b64(640, 480, (i, 0, 0))} for i in range(2)],
    # All modalities in one input (1035 MB).
    "mixed_8img_30s_8f": lambda: [
        {"text": "mixed", "image": [png_b64(1024, 768)] * 8, "audio": wav_b64(30.0, 48000), "video": frames_b64(8, 1280, 720)}
    ],
}


def planned_pass_estimates(inputs: list[dict], count_text_tokens) -> list[float]:
    """Estimates of the passes the server would plan for this request (CPU: decode, token formula, chunker)."""
    limits = MediaLimits.from_settings(normalize_settings(GPU_SETTINGS))
    items = [
        MediaInputData(
            text=item.get("text"),
            image=api._as_list(item.get("image")),
            audio=api._as_list(item.get("audio")),
            video=item.get("video"),
        )
        for item in inputs
    ]
    decoded = decode_inputs(items, limits)
    costs = [item_cost(item, input_tokens(count_text_tokens(item.text or ""), item)) for item in decoded]
    assert all(cost.tokens <= 4096 for cost in costs), [cost.tokens for cost in costs]
    return [estimate_pass_mb([costs[index] for index in chunk], COST_MB) for chunk in plan_chunks(costs, BUDGET_MB, COST_MB)]


def test_gpu_switch_and_documented_command(monkeypatch):
    monkeypatch.setenv("G3_EMBED_GPU_TESTS", "1 ")  # what cmd.exe stores for `set G3_EMBED_GPU_TESTS=1 && ...`
    assert gpu_tests_enabled()
    monkeypatch.setenv("G3_EMBED_GPU_TESTS", "0")
    assert not gpu_tests_enabled()
    assert 'set "G3_EMBED_GPU_TESTS=1" &&' in __doc__


@pytest.fixture(scope="module")
def count_text_tokens():
    if model_files_cached() is None:
        pytest.skip(f"{MODEL} not in models/")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_files_cached()))
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0


@pytest.mark.parametrize("case", sorted(CASES))
def test_cases_plan_near_budget(count_text_tokens, case):
    estimates = planned_pass_estimates(CASES[case](), count_text_tokens)
    print(f"\n{case}: {[round(value) for value in estimates]}")
    assert max(estimates) >= NEAR_BUDGET * BUDGET_MB, f"{case}: largest planned pass {max(estimates):.0f} MB"
    assert max(estimates) <= BUDGET_MB


@pytest.fixture(scope="module")
def loaded_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if model_files_cached() is None:
        pytest.skip(f"{MODEL} not in models/")
    free, total = torch.cuda.mem_get_info()
    if free < 3 * GIB:
        pytest.skip(f"only {free / GIB:.1f} GiB VRAM free, need 3 GiB")
    torch.cuda.set_per_process_memory_fraction(min(1.0, PROCESS_CAP_BYTES / total))
    settings = normalize_settings(GPU_SETTINGS)
    runtime_key, choice, snapshot = runtime.build_runtime_key(MODEL, settings)
    assert choice.backend == "pytorch" and choice.device.startswith("cuda"), choice
    with runtime._model_cache_lock:
        runtime._load_model(runtime_key, snapshot, choice)
        entry = runtime._model_cache[runtime_key]
    runtime._release_cuda_cache()
    print(
        f"\n[gpu] loaded: allocated {torch.cuda.memory_allocated() / MIB:.0f} MB, "
        f"reserved {torch.cuda.memory_reserved() / MIB:.0f} MB, cap {PROCESS_CAP_BYTES / MIB:.0f} MB"
    )
    yield runtime_key, entry
    runtime._model_cache.pop(runtime_key, None)
    entry.clear()
    runtime._release_cuda_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)


@pytest.fixture
def gpu_client(loaded_model):
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient

    runtime_key, entry = loaded_model
    runtime._model_cache[runtime_key] = entry  # the autouse fixture gives every test an empty cache
    current_settings.clear()
    current_settings.update(normalize_settings(GPU_SETTINGS))
    app = FastAPI()
    app.add_middleware(api.RequestBodyLimitMiddleware)
    app.add_exception_handler(RequestValidationError, api.request_validation_exception_handler)
    api.create_api(app)
    with TestClient(app) as client:
        yield client


def measure(client, inputs, monkeypatch):
    import torch

    planned: list[float] = []
    real_plan_chunks = runtime.plan_chunks

    def recording_plan_chunks(costs, budget_mb, cost_mb):
        chunks = real_plan_chunks(costs, budget_mb, cost_mb)
        planned.extend(estimate_pass_mb([costs[index] for index in chunk], cost_mb) for chunk in chunks)
        return chunks

    monkeypatch.setattr(runtime, "plan_chunks", recording_plan_chunks)
    runtime._release_cuda_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    response = client.post("/embed", json={"model": MODEL, "inputs": inputs})
    torch.cuda.synchronize()
    peak_extra_mb = (torch.cuda.max_memory_allocated() - baseline) / MIB
    reserved_after_mb = torch.cuda.memory_reserved() / MIB
    return response, planned, peak_extra_mb, reserved_before / MIB, reserved_after_mb


@gpu_only
@pytest.mark.parametrize("case", sorted(CASES))
def test_near_budget_media_stays_within_budget(gpu_client, monkeypatch, case):
    inputs = CASES[case]()
    response, planned, peak_mb, reserved_before_mb, reserved_after_mb = measure(gpu_client, inputs, monkeypatch)
    body = response.json()
    print(
        f"\n[gpu] {case}: status {response.status_code}, tokens {body.get('input_tokens')}, media {body.get('media')}, "
        f"passes {[round(value) for value in planned]} MB planned, peak extra {peak_mb:.0f} MB (budget {BUDGET_MB}), "
        f"reserved {reserved_before_mb:.0f} -> {reserved_after_mb:.0f} MB"
    )
    assert response.status_code == 200, body
    assert len(body["vectors"]) == len(inputs) and body["dimension"] == 768
    assert planned and max(planned) >= NEAR_BUDGET * BUDGET_MB, f"{case}: no pass near the budget ({planned})"
    assert peak_mb <= BUDGET_MB, f"{case}: peak {peak_mb:.0f} MB > budget {BUDGET_MB} MB"
    assert reserved_after_mb <= reserved_before_mb + RELEASE_SLACK_MB, "reserved memory not released"
