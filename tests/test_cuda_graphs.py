"""CUDA graphs for short EmbeddingGemma 2 text passes, the to()/eval() fast path and the runtime wiring.

CPU tests cover buckets, model support checks and settings. The opt-in GPU test (G3_EMBED_GPU_TESTS=1, model
files cached, ~2 GiB free VRAM) captures real graphs and compares them with the eager forward:

    set "G3_EMBED_GPU_TESTS=1" && venv\\Scripts\\python.exe -m pytest tests/test_cuda_graphs.py -q -s
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from backend import genesis_embed_server_cuda_graphs as cuda_graphs
from backend import genesis_embed_server_runtime as runtime
from backend.genesis_embed_server_hardware import RuntimeChoice
from backend.genesis_embed_server_storage import DEFAULT_SETTINGS, normalize_settings
from conftest import PROJECT_ROOT, gpu_tests_enabled, model_files_cached

GEMMA = "google/embeddinggemma-2"


def test_bucket_for():
    assert cuda_graphs.bucket_for(1, (16, 32)) == 16
    assert cuda_graphs.bucket_for(16, (16, 32)) == 16
    assert cuda_graphs.bucket_for(17, (16, 32)) == 32
    assert cuda_graphs.bucket_for(33, (16, 32)) is None


def test_graph_buckets_largest_first_and_capped():
    buckets = cuda_graphs.graph_buckets(4096)
    sizes = [rows * tokens for rows, tokens in buckets]
    assert sizes == sorted(sizes, reverse=True)
    assert max(sizes) <= 4096 and (1, 16) in buckets and (128, 32) in buckets and (8, 512) in buckets
    assert (16, 512) not in buckets
    assert len(buckets) == len(set(buckets))


def test_pool_reserved_bytes_counts_only_the_graph_pool_on_its_device():
    segments = [
        {"device": 0, "segment_pool_id": (0, 0), "total_size": 900 << 20},  # ordinary cache (e.g. a compile warmup)
        {"device": 0, "segment_pool_id": (1, 7), "total_size": 400 << 20},
        {"device": 0, "segment_pool_id": (1, 7), "total_size": 250 << 20},
        {"device": 1, "segment_pool_id": (1, 7), "total_size": 64 << 20},
        {"device": 0, "segment_pool_id": (1, 8), "total_size": 32 << 20},
    ]
    assert cuda_graphs.pool_reserved_bytes(segments, (1, 7), 0) == 650 << 20
    assert cuda_graphs.pool_reserved_bytes(segments, (1, 7), None) == 714 << 20
    assert cuda_graphs.pool_reserved_bytes([], (1, 7), 0) == 0
    # A snapshot without pool ids: the caller falls back to the reserved-memory delta.
    assert cuda_graphs.pool_reserved_bytes([{"device": 0, "total_size": 1}], (1, 7), 0) is None


def test_supported_rejects_non_gemma_and_cpu():
    ok, reason = cuda_graphs.supported(torch.nn.Linear(2, 2))
    assert not ok and reason


def test_settings_defaults_and_payload():
    assert DEFAULT_SETTINGS["cuda_graphs"] is True and DEFAULT_SETTINGS["encode_token_batching"] is True
    normalized = normalize_settings({"cuda_graphs": False, "encode_token_batching": 0})
    assert normalized["cuda_graphs"] is False and normalized["encode_token_batching"] is False
    assert normalize_settings({})["cuda_graphs"] is True


@pytest.mark.parametrize(
    "device, model_id, enabled, expected",
    [
        ("cuda", GEMMA, True, True),
        ("cuda", GEMMA, False, False),
        ("cpu", GEMMA, True, False),
        ("cuda", "intfloat/multilingual-e5-large", True, False),
    ],
)
def test_runtime_key_cuda_graphs(monkeypatch, device, model_id, enabled, expected):
    choice = RuntimeChoice(backend="pytorch", device=device, device_label=device, source="test", detail="test")
    monkeypatch.setattr(runtime, "resolve_runtime_choice", lambda settings: choice)
    key, _, _ = runtime.build_runtime_key(model_id, {"cuda_graphs": enabled})
    assert key.cuda_graphs is expected


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.to_calls = 0
        self.eval_calls = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        self.to_calls += 1
        return super().to(*args, **kwargs)

    def eval(self):
        self.eval_calls += 1
        return super().eval()


def test_skip_redundant_to_and_eval():
    model = _Model()
    runtime._skip_redundant_to_and_eval(model)
    assert model.eval() is model and model.eval_calls == 1  # training -> real eval()
    assert model.eval() is model and model.eval_calls == 1  # already eval -> skipped
    assert model.to("cpu") is model and model.to(torch.device("cpu")) is model and model.to_calls == 0
    model.to(torch.float64)
    assert model.to_calls == 1 and model.layer.weight.dtype == torch.float64
    model.train()
    model.eval()
    assert model.eval_calls == 2 and not model.layer.training


def test_runtime_status_reports_graphs(monkeypatch):
    class Runner:
        def status(self):
            return {"graphs": 3}

    key = runtime.RuntimeKey(GEMMA, "pytorch", "cuda", "bfloat16", "full", cuda_graphs=True)
    monkeypatch.setitem(runtime._model_cache, key, {"model": None, "cuda_graphs": Runner()})
    monkeypatch.setattr(runtime, "get_system_snapshot", lambda: {})
    loaded = runtime.get_runtime_status()["loaded_models"]
    assert loaded[0]["cuda_graphs"] == {"graphs": 3}


# --- GPU -----------------------------------------------------------------------------------------------------

gpu_only = pytest.mark.skipif(
    not gpu_tests_enabled() or not torch.cuda.is_available() or model_files_cached() is None,
    reason="set G3_EMBED_GPU_TESTS=1 (CUDA and cached model files needed)",
)


@pytest.fixture(scope="module")
def gemma():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        GEMMA,
        device="cuda",
        cache_folder=str(PROJECT_ROOT / "models"),
        model_kwargs={"dtype": torch.bfloat16},
    )
    yield model
    del model
    runtime._release_cuda_cache()


def _encode(model, texts, batch_size):
    return model.encode(
        [{"text": text} for text in texts], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )


@gpu_only
def test_graphs_match_eager(gemma):
    texts = [("Rechnung Vertrag Kunde " * (index % 9 + 1)).strip() for index in range(11)]
    texts.append("long text " * 400)  # > 512 tokens: eager fallback
    eager = {size: _encode(gemma, texts, size) for size in (1, 4, 12)}
    inner = gemma._first_module().model
    runner = cuda_graphs.TextGraphRunner(inner, max_tokens=1024)
    runner.capture([(16, 64), (4, 32), (1, 16), (4, 64), (1, 64)])
    runner.install()
    try:
        # memory_bytes is the graph pool's own segments, so it is never 0 once a graph is captured.
        assert runner.memory_bytes > 0 and runner.memory_bytes == runner.pool_bytes() and len(runner.graphs) == 5
        for size in (1, 4, 12):
            graphed = _encode(gemma, texts, size)
            # Normalized in bf16 by sentence-transformers (norms off by up to ~0.5%): renormalize first.
            a = graphed.astype("float64") / np.linalg.norm(graphed.astype("float64"), axis=1, keepdims=True)
            b = eager[size].astype("float64") / np.linalg.norm(eager[size].astype("float64"), axis=1, keepdims=True)
            cosine = (a * b).sum(axis=1)
            assert cosine.min() > 0.9999, (size, cosine.min())
        assert runner.replays > 0 and runner.fallbacks > 0
    finally:
        runner.uninstall()
    assert "forward" not in inner.__dict__
