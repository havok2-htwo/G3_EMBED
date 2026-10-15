"""Runtime encode paths against a FakeModel: text wrapping, warmups, chunking, token counts, compile targets."""
from __future__ import annotations

import gc
import importlib.util
import weakref

import numpy as np
import pytest
import torch
from PIL import Image

from backend import genesis_embed_server_runtime as runtime
from backend.genesis_embed_server_hardware import RuntimeChoice
from backend.genesis_embed_server_media import DecodedMedia, image_tokens

GEMMA = "google/embeddinggemma-2"


def image_input(count=1, text=None):
    return DecodedMedia(text=text, images=[Image.new("RGB", (64, 48))] * count)


def test_encode_texts_wraps_every_string(fake_model):
    result = runtime.encode_texts(GEMMA, ["https://x/y.jpg", "C:\\x.png", ""])
    assert fake_model.calls[0]["inputs"] == [{"text": "https://x/y.jpg"}, {"text": "C:\\x.png"}, {"text": ""}]
    assert result.input_count == 3 and result.input_tokens is None and result.media is None


def test_compiled_warmup_wraps_texts(fake_model):
    runtime._warmup_compiled_model(fake_model, 4096, 2)
    assert fake_model.calls
    for call in fake_model.calls:
        assert all(isinstance(item, dict) and list(item) == ["text"] for item in call["inputs"])


def test_count_input_tokens_mixed(fake_model):
    counted = runtime.count_input_tokens(GEMMA, ["one two", image_input(2, text="a b c")])
    assert counted.model_id == GEMMA and counted.max_tokens == 4096
    assert counted.counts == [2 + 2, 2 + 3 + 2 * image_tokens(48, 64)]


def test_validate_input_lengths_still_raises(fake_model, settings):
    settings(max_seq_length=5)
    assert runtime.validate_input_lengths(GEMMA, ["short"]) == 0.0
    with pytest.raises(runtime.InputTooLongError) as error:
        runtime.validate_input_lengths(GEMMA, ["short", "a b c d e f"])
    assert error.value.too_long == [{"index": 1, "tokens": 8}]


def test_encode_inputs_order_chunks_and_summary(fake_model, settings):
    settings(media_pass_budget_mb=300)
    inputs = ["alpha", image_input(3), "be", image_input(1), image_input(1), "gamma"]
    tokens = [3, 900, 3, 300, 300, 3]
    result = runtime.encode_inputs(GEMMA, inputs, None, tokens)
    assert [row[0] for row in result.vectors] == [5, 3000, 2, 1000, 1000, 5]
    media_calls = fake_model.media_calls()
    # 3 images = 272 MB; adding one more image would be 346 > 300. The two single images share a pass (198 MB).
    assert [call["batch_size"] for call in media_calls] == [1, 2]
    text_call = fake_model.calls[0]
    assert text_call["inputs"] == [{"text": "alpha"}, {"text": "be"}, {"text": "gamma"}]
    assert result.input_tokens == sum(tokens)
    assert result.media == {"images": 5, "audio_seconds": 0.0, "video_frames": 0}
    assert result.input_chars == len("alpha") + len("be") + len("gamma")


def test_encode_inputs_releases_cache_after_cuda_media(fake_model, monkeypatch):
    released = []
    monkeypatch.setattr(runtime, "_is_cuda_pytorch", lambda choice: True)
    monkeypatch.setattr(runtime, "_check_cuda_headroom", lambda required_mb, device: None)
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: released.append("gc"))
    monkeypatch.setattr(runtime, "_empty_cuda_cache", lambda: released.append("empty"))
    runtime.encode_inputs(GEMMA, [image_input()], None, [300])
    # A successful pass skips gc.collect() (~0.3-1 s with torch.compile state) and only empties the cache.
    assert released == ["empty"]
    runtime.encode_inputs(GEMMA, ["text only"], None, [3])
    assert released == ["empty"]  # text-only calls keep the cache


# --- token-budget batching ----------------------------------------------------------------------------------

def test_plan_token_batches_packs_by_padded_tokens():
    lengths = [10, 4096, 12, 4000, 8, 11]
    batches = runtime.plan_token_batches(lengths, token_budget=8192, max_rows=3)
    # Longest first; each pass is padded to its first entry: 2 x 4096 <= 8192, then 3 rows (cap) of <= 12.
    assert batches == [[1, 3], [2, 5, 0], [4]]
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) * max(lengths[index] for index in batch) <= 8192


def test_plan_token_batches_oversized_entry_gets_own_pass():
    assert runtime.plan_token_batches([9000, 5, 0], token_budget=4096, max_rows=128) == [[0], [1, 2]]
    assert runtime.plan_token_batches([], token_budget=4096) == []


def test_encode_texts_packs_short_texts_into_one_pass(fake_model):
    texts = [" ".join(["w"] * (index % 7 + 1)) for index in range(40)]
    result = runtime.encode_texts(GEMMA, texts)
    # Default EmbeddingGemma 2 budget: 4 x 4096 tokens, so 40 short texts share one pass instead of 10 passes of 4.
    assert [call["batch_size"] for call in fake_model.calls] == [40]
    assert [row[0] for row in result.vectors] == [float(len(text)) for text in texts]


def test_encode_texts_token_budget_follows_batch_size_and_window(fake_model, settings):
    settings(encode_batch_size=2, max_seq_length=8)
    texts = ["a b c d e f", "a", "a b", "a b c d e f g h i j", "a b c"]  # tokens incl. <bos>/<eos>: 8 3 4 12 5
    result = runtime.encode_texts(GEMMA, texts)
    # Budget 2 x 8 = 16 padded tokens; lengths are capped at the window (12 -> 8, a tie kept in input order).
    assert [[item["text"] for item in call["inputs"]] for call in fake_model.calls] == [
        ["a b c d e f", "a b c d e f g h i j"],
        ["a b c", "a b", "a"],
    ]
    assert [row[0] for row in result.vectors] == [float(len(text)) for text in texts]


def test_encode_texts_token_batching_can_be_disabled(fake_model, settings):
    settings(encode_token_batching=False)
    runtime.encode_texts(GEMMA, ["x"] * 10)
    assert [call["batch_size"] for call in fake_model.calls] == [4]  # one encode call, fixed batch size


def test_encode_inputs_text_part_uses_token_batching(fake_model):
    texts = ["word"] * 9
    result = runtime.encode_inputs(GEMMA, texts + [image_input()], None, [3] * 9 + [300])
    assert [call["batch_size"] for call in fake_model.calls] == [9, 1]
    assert [row[0] for row in result.vectors] == [4.0] * 9 + [1000.0]


def test_encode_inputs_headroom_checked_per_chunk(fake_model, monkeypatch, settings):
    settings(media_pass_budget_mb=200)
    checks = []
    monkeypatch.setattr(runtime, "_is_cuda_pytorch", lambda choice: True)
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr(runtime, "_check_cuda_headroom", lambda required_mb, device: checks.append(required_mb))
    runtime.encode_inputs(GEMMA, [image_input(2), image_input(2)], None, [600, 600])
    assert checks == [pytest.approx(2 * 74 + 50)] * 2


def test_encode_texts_oom_raises_typed_error(fake_model, monkeypatch):
    released = []
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: released.append(True))
    fake_model.fail_with = torch.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(runtime.GpuOutOfMemory):
        runtime.encode_texts(GEMMA, ["x"])
    assert released == [True]
    fake_model.fail_with = ValueError("other")
    with pytest.raises(ValueError):
        runtime.encode_texts(GEMMA, ["x"])


def test_encode_inputs_frees_failed_pass_before_releasing_cache(fake_model, monkeypatch):
    # A non-OOM error mid-pass: the cache release must run after the traceback (whose frames hold the pass's
    # tensors) is gone, else empty_cache cannot return that memory to the other services.
    class Tensor:
        pass

    refs = []
    alive_at_release = []

    def failing_encode(inputs, batch_size=32, **kwargs):
        features = Tensor()
        refs.append(weakref.ref(features))
        try:
            raise KeyError("inner")
        except KeyError:
            raise RuntimeError("forward failed")

    def release():
        gc.collect()
        alive_at_release.append(refs[-1]() is not None)

    monkeypatch.setattr(fake_model, "encode", failing_encode)
    monkeypatch.setattr(runtime, "_is_cuda_pytorch", lambda choice: True)
    monkeypatch.setattr(runtime, "_check_cuda_headroom", lambda required_mb, device: None)
    monkeypatch.setattr(runtime, "_release_cuda_cache", release)
    with pytest.raises(RuntimeError, match="forward failed"):
        runtime.encode_inputs(GEMMA, [image_input()], None, [300])
    assert alive_at_release == [False]


@pytest.mark.parametrize("message", ["CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"])
def test_encode_inputs_allocation_failures_are_oom(fake_model, monkeypatch, message):
    released = []
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: released.append(True))
    fake_model.fail_with = RuntimeError(message)
    with pytest.raises(runtime.GpuOutOfMemory):
        runtime.encode_inputs(GEMMA, [image_input()], None, [300])
    assert released == [True]


def test_encode_inputs_headroom_counts_padded_audio(fake_model, monkeypatch):
    checks = []
    monkeypatch.setattr(runtime, "_is_cuda_pytorch", lambda choice: True)
    monkeypatch.setattr(runtime, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr(runtime, "_check_cuda_headroom", lambda required_mb, device: checks.append(required_mb))
    short = DecodedMedia(audio=[np.zeros(1600, np.float32)] * 2)
    inputs = [DecodedMedia(audio=[np.zeros(30 * 16000, np.float32)])] + [short] * 5
    runtime.encode_inputs(GEMMA, inputs, None, [754] + [12] * 5)
    # 1 + 4 x 2 clips, all padded to 30 s: 9 x 30 s x 3.5 MB; a fifth short input (11 clips) would exceed 1200 MB.
    assert [call["batch_size"] for call in fake_model.media_calls()] == [5, 1]
    assert checks == [pytest.approx(9 * 30 * 3.5 + 50), pytest.approx(0.125 * 12 + 50)]


def test_check_cuda_headroom(monkeypatch):
    mib = 1024 * 1024
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1000 * mib, 24000 * mib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 500 * mib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 300 * mib)
    runtime._check_cuda_headroom(1200 - 256, "cuda")  # 1000 free + 200 cached-unused = 1200
    with pytest.raises(runtime.GpuMemoryUnavailable) as error:
        runtime._check_cuda_headroom(1000, "cuda")
    assert error.value.required_mb == 1256 and error.value.available_mb == 1200


# --- torch.compile targets ----------------------------------------------------------------------------------

class _Inner(torch.nn.Module):
    def __init__(self, towers: bool):
        super().__init__()
        self.language_model = torch.nn.Linear(2, 2)
        self.vision_tower = torch.nn.Linear(2, 2) if towers else None
        self.audio_tower = torch.nn.Linear(2, 2) if towers else None


class _FirstModule:
    def __init__(self, inner):
        self.model = inner


class _FakeST:
    def __init__(self, inner):
        self.first = _FirstModule(inner)

    def _first_module(self):
        return self.first


@pytest.fixture
def compile_env(monkeypatch):
    compiled = {"modules": [], "full": []}
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "triton" else real_find_spec(name, *a))
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, *a, **k: compiled["modules"].append(self))

    def fake_compile(module, *args, **kwargs):
        compiled["full"].append(module)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    return compiled


def test_full_compile_skips_media_towers(compile_env):
    inner = _Inner(towers=True)
    choice = RuntimeChoice("pytorch", "cuda", "GPU", "test", "test")
    assert runtime._maybe_compile_sentence_transformer(_FakeST(inner), "full", choice)
    assert compile_env["modules"] == [inner.language_model] and compile_env["full"] == []


def test_full_compile_text_model_unchanged(compile_env):
    inner = _Inner(towers=False)
    model = _FakeST(inner)
    choice = RuntimeChoice("pytorch", "cuda", "GPU", "test", "test")
    assert runtime._maybe_compile_sentence_transformer(model, "full", choice)
    assert compile_env["full"] == [inner] and compile_env["modules"] == []


def test_as_rows_reshapes_single_vector():
    assert runtime._as_rows(np.zeros(4)).shape == (1, 4)


def test_token_counting_uses_the_model_tokenizer_under_the_tokenizer_lock(fake_model, monkeypatch):
    """No private tokenizer copy: counts come from the shared tokenizer, serialized with encode()'s preprocess."""
    held = []
    original_call = type(fake_model.tokenizer).__call__

    def call(self, *args, **kwargs):
        lock = next(iter(runtime._model_cache.values()))["tokenizer_lock"]
        held.append(lock._is_owned())
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(type(fake_model.tokenizer), "__call__", call)
    counts = runtime.count_input_tokens(GEMMA, ["a b c", "d"])
    assert counts.counts == [5, 3]
    assert held == [True]
    assert all(not key.startswith("count_") for key in next(iter(runtime._model_cache.values())))


def test_serialize_preprocess_holds_the_lock_during_tokenization():
    import threading

    lock = threading.RLock()
    seen = []

    class Model:
        def preprocess(self, inputs, prompt=None):
            seen.append((lock._is_owned(), inputs, prompt))
            return {"input_ids": inputs}

    model = Model()
    runtime._serialize_preprocess(model, lock)
    assert model.preprocess(["x"], prompt="p") == {"input_ids": ["x"]}
    assert seen == [(True, ["x"], "p")]
    assert not lock._is_owned()
