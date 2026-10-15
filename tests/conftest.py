"""Shared fixtures: isolated settings/logs, a fake SentenceTransformer, and a TestClient on the public API.

Nothing here loads a real model, touches the GPU, or writes to logs/.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_REAL_LOGS_DIR = (PROJECT_ROOT / "logs").resolve()
_real_write_text = Path.write_text


def _write_text_outside_real_logs(self, *args, **kwargs):
    if Path(self).resolve().parent == _REAL_LOGS_DIR:
        return 0
    return _real_write_text(self, *args, **kwargs)


# Importing the backend creates AuthStore(PROJECT_ROOT), which rewrites logs/genesis_embed_secrets.json.
# Tests never write the real logs/, so drop writes there while the backend is imported.
Path.write_text = _write_text_outside_real_logs
try:
    from backend import genesis_embed_server_auth as auth  # noqa: E402
finally:
    Path.write_text = _real_write_text

from backend import genesis_embed_server_api as api  # noqa: E402
from backend import genesis_embed_server_runtime as runtime  # noqa: E402
from backend import genesis_embed_server_storage as storage  # noqa: E402
from backend.genesis_embed_server_globals import current_settings, embedding_history  # noqa: E402
from backend.genesis_embed_server_hardware import RuntimeChoice  # noqa: E402

FAKE_DIMENSION = 8


class FakeTokenizer:
    """One token per whitespace-separated word, plus <bos>/<eos> with add_special_tokens."""

    def __call__(self, texts, add_special_tokens=True, truncation=False, verbose=False):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        ids = [[1] * (len(text.split()) + (2 if add_special_tokens else 0)) for text in batch]
        return {"input_ids": ids[0] if single else ids}


def input_tag(item) -> float:
    """Distinct number per input so tests can check the order of returned vectors."""
    if isinstance(item, str):
        return float(len(item))
    text = item.get("text", "")
    return float(
        1000 * len(item.get("image", []))
        + 100 * len(item.get("audio", []))
        + 10 * len(item.get("video", []))
        + len(text)
    )


class FakeModel:
    def __init__(self):
        self.calls: list[dict] = []
        self.max_seq_length = 8192
        self.tokenizer = FakeTokenizer()
        self.fail_with: BaseException | None = None

    def encode(self, inputs, batch_size=32, **kwargs):
        inputs = list(inputs)
        self.calls.append({"inputs": inputs, "batch_size": batch_size})
        if self.fail_with is not None:
            raise self.fail_with
        vectors = np.zeros((len(inputs), FAKE_DIMENSION), dtype=np.float32)
        for row, item in enumerate(inputs):
            vectors[row, 0] = input_tag(item)
        return vectors

    def media_calls(self) -> list[dict]:
        return [call for call in self.calls if any(set(item) - {"text"} for item in call["inputs"] if isinstance(item, dict))]


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Default settings, logs in tmp_path, no auth, empty history and model cache for every test."""
    monkeypatch.setattr(storage, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(storage, "SETTINGS_FILE", tmp_path / "genesis_embed_settings.json")
    monkeypatch.setattr(storage, "LOG_FILE", tmp_path / "embedding_log.jsonl")
    monkeypatch.setattr(auth.get_auth_store(), "secrets_path", tmp_path / "genesis_embed_secrets.json")
    logged: list[dict] = []
    monkeypatch.setattr(api, "log_embedding", logged.append)
    monkeypatch.setattr(api, "authorize_api_key", lambda request: None)
    monkeypatch.setattr(api, "media_slots", api._MediaSlots())
    monkeypatch.setattr(runtime, "_model_cache", {})
    monkeypatch.setattr(runtime, "_standalone_tokenizers", runtime.OrderedDict())
    saved = dict(current_settings)
    current_settings.clear()
    current_settings.update(storage.normalize_settings({}))
    embedding_history.clear()
    yield logged
    embedding_history.clear()
    current_settings.clear()
    current_settings.update(saved)


@pytest.fixture
def logged(isolated_state):
    return isolated_state


@pytest.fixture
def settings():
    """Change live settings for one test: settings(max_video_frames=2, ...)."""

    def apply(**values):
        current_settings.update(storage.normalize_settings({**current_settings, **values}))
        return dict(current_settings)

    return apply


@pytest.fixture
def fake_model(monkeypatch):
    """Route every model load to one FakeModel on a PyTorch CPU runtime (no hardware probes)."""
    model = FakeModel()
    choice = RuntimeChoice(backend="pytorch", device="cpu", device_label="CPU", source="test", detail="test")
    monkeypatch.setattr(runtime, "resolve_runtime_choice", lambda settings: choice)

    def load(runtime_key, settings, runtime_choice):
        if runtime_key not in runtime._model_cache:
            runtime._model_cache[runtime_key] = {
                "model": model,
                "loaded_at": 0.0,
                "load_duration_ms": 0.0,
                "choice": runtime_choice,
                "native_max_seq_length": 8192,
                "encode_lock": threading.Lock(),
                "tokenizer_lock": threading.RLock(),
            }
        return model, 0.0

    monkeypatch.setattr(runtime, "_load_model", load)
    # Tokenizer-only token checks (models not loaded yet) count with the same fake tokenizer.
    model.tokenizer_loads = []

    def load_tokenizer(model_id, cache_folder):
        model.tokenizer_loads.append(model_id)
        return FakeTokenizer()

    monkeypatch.setattr(runtime, "_load_standalone_tokenizer", load_tokenizer)
    return model


@pytest.fixture
def client(fake_model):
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.add_middleware(api.RequestBodyLimitMiddleware)
    app.add_exception_handler(RequestValidationError, api.request_validation_exception_handler)
    api.create_api(app)
    with TestClient(app) as test_client:
        yield test_client


def model_files_cached(model_dir_name: str = "models--google--embeddinggemma-2") -> Path | None:
    snapshots = PROJECT_ROOT / "models" / model_dir_name / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir()):
        if (snapshot / "processor_config.json").exists() or (snapshot / "preprocessor_config.json").exists():
            return snapshot
    return None


def gpu_tests_enabled() -> bool:
    # strip(): cmd.exe `set G3_EMBED_GPU_TESTS=1 && ...` stores "1 " (the space before &&).
    return os.environ.get("G3_EMBED_GPU_TESTS", "").strip() == "1"
