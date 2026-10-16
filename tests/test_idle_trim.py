"""Idle trim: cached CUDA memory goes back to the driver idle_trim_seconds after the last encode call.

CPU only: the torch.cuda calls of trim_memory are replaced, the trim thread runs for real.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

from backend import genesis_embed_server_runtime as runtime
from backend.genesis_embed_server_hardware import RuntimeChoice
from backend.genesis_embed_server_storage import DEFAULT_SETTINGS, normalize_settings

GEMMA = "google/embeddinggemma-2"


@pytest.fixture
def idle(monkeypatch):
    """Fresh idle state and trim stats; trim_memory records its calls instead of touching CUDA."""
    monkeypatch.setattr(runtime, "_idle_state", {"busy": 0, "last_activity": time.monotonic(), "pending": False, "thread": None})
    monkeypatch.setattr(runtime, "_trim_stats", {"trims": 0, "last_trim_at": None, "last_freed_mb": None})
    monkeypatch.setattr(runtime, "_IDLE_TRIM_POLL_SECONDS", 0.05)
    trims: list[float] = []
    monkeypatch.setattr(runtime, "trim_memory", lambda: trims.append(time.monotonic()) or {"trimmed": True})
    yield trims
    with runtime._idle_cond:
        runtime._idle_state["thread"] = None  # lets the worker of this test exit
        runtime._idle_cond.notify_all()


def _wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_setting_default_range_and_preset_key():
    from backend import genesis_embed_server_presets as presets

    assert DEFAULT_SETTINGS["idle_trim_seconds"] == 5
    assert normalize_settings({})["idle_trim_seconds"] == 5
    assert normalize_settings({"idle_trim_seconds": -3})["idle_trim_seconds"] == 0
    assert normalize_settings({"idle_trim_seconds": 99999})["idle_trim_seconds"] == 3600
    assert "idle_trim_seconds" in presets.PRESET_KEYS


def test_trim_runs_once_after_the_idle_time(idle, settings):
    settings(idle_trim_seconds=1)
    started = time.monotonic()
    runtime._begin_activity()
    runtime._end_activity(True)
    assert runtime.idle_trim_status()["pending"] is True
    assert _wait_for(lambda: idle), "no trim within 3 s"
    assert idle[0] - started >= 0.95
    time.sleep(0.2)
    assert len(idle) == 1, "one trim per idle period"
    assert runtime.idle_trim_status()["pending"] is False


def test_no_trim_while_a_pass_is_running(idle, settings):
    settings(idle_trim_seconds=1)
    runtime._begin_activity()
    runtime._end_activity(True)
    runtime._begin_activity()  # a second request is still running
    time.sleep(1.3)
    assert idle == []
    assert runtime.idle_trim_status()["next_trim_in_seconds"] is None
    runtime._end_activity(True)
    assert _wait_for(lambda: idle)


def test_new_activity_restarts_the_idle_time(idle, settings):
    settings(idle_trim_seconds=1)
    runtime._begin_activity()
    runtime._end_activity(True)
    time.sleep(0.6)
    runtime._begin_activity()
    last = time.monotonic()
    runtime._end_activity(True)
    assert _wait_for(lambda: idle)
    assert idle[0] - last >= 0.95


def test_cpu_passes_and_zero_seconds_never_trim(idle, settings):
    runtime._begin_activity()
    runtime._end_activity(False)
    assert runtime._idle_state["pending"] is False and runtime._idle_state["thread"] is None
    settings(idle_trim_seconds=0)
    runtime._begin_activity()
    runtime._end_activity(True)
    assert runtime._idle_state["pending"] is False
    assert runtime.idle_trim_status()["idle_trim_seconds"] == 0


def test_encode_texts_counts_as_activity(idle, fake_model, monkeypatch):
    seen = []
    monkeypatch.setattr(runtime, "_end_activity", lambda on_cuda: seen.append((runtime._idle_state["busy"], on_cuda)))
    runtime.encode_texts(GEMMA, ["a b"])
    runtime.encode_inputs(GEMMA, ["a b"])
    assert seen == [(1, False), (2, False)], "the fake model runs on CPU, so no trim is scheduled"
    fake_model.fail_with = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        runtime.encode_texts(GEMMA, ["a b"])
    assert seen[-1] == (3, True), "a failed call schedules a trim (it may have run passes on CUDA)"


def test_trim_memory_holds_every_encode_lock(monkeypatch):
    calls = []
    entries = {}
    for index in range(2):
        key = runtime.RuntimeKey(f"m{index}", "pytorch", "cuda:0", "auto", "off")
        entries[key] = {"choice": RuntimeChoice("pytorch", "cuda:0", "GPU", "test", "test"), "encode_lock": threading.Lock()}
    monkeypatch.setattr(runtime, "_model_cache", entries)
    monkeypatch.setattr(runtime, "_trim_stats", {"trims": 0, "last_trim_at": None, "last_freed_mb": None})
    monkeypatch.setattr(runtime, "_bind_cuda_device", lambda device: calls.append(("bind", device)))
    reserved = iter([3000 * 1024 * 1024, 900 * 1024 * 1024])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: next(reserved))

    def empty_cache():
        calls.append(("empty", [entry["encode_lock"].locked() for entry in entries.values()]))

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    result = runtime.trim_memory()
    assert calls == [("bind", "cuda:0"), ("empty", [True, True])]
    assert result == {"trimmed": True, "reserved_before_mb": 3000.0, "reserved_after_mb": 900.0, "freed_mb": 2100.0}
    assert not any(entry["encode_lock"].locked() for entry in entries.values())
    assert runtime._trim_stats["trims"] == 1 and runtime._trim_stats["last_freed_mb"] == 2100.0


def test_trim_memory_without_cuda_is_a_no_op(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("empty_cache without CUDA"))
    assert runtime.trim_memory()["trimmed"] is False


def test_runtime_status_reports_idle_trim(monkeypatch):
    monkeypatch.setattr(runtime, "get_system_snapshot", lambda: {})
    status = runtime.get_runtime_status()["idle_trim"]
    assert status["idle_trim_seconds"] == 5 and set(status) >= {"pending", "next_trim_in_seconds", "trims", "last_freed_mb"}
