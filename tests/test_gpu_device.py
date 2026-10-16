"""GPU selection: the gpu_device setting, its dropdown options and the CUDA device the runtime choice resolves to."""
from __future__ import annotations

import pytest

from backend import genesis_embed_server_hardware as hardware
from backend.genesis_embed_server_storage import DEFAULT_SETTINGS, normalize_settings

TWO_CARDS = {
    "torch": {
        "cuda_available": True,
        "cuda_devices": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 2070 SUPER"],
        "cuda_memory_mib": [24576, 8192],
        "xpu_available": False,
        "xpu_devices": [],
    },
    "openvino_backend_available": False,
    "openvino_devices": [],
}


@pytest.fixture
def two_cards(monkeypatch):
    monkeypatch.setattr(hardware, "get_system_snapshot", lambda: TWO_CARDS)


def test_default_is_auto():
    assert DEFAULT_SETTINGS["gpu_device"] == "auto"
    assert normalize_settings({})["gpu_device"] == "auto"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "auto"), ("", "auto"), ("auto", "auto"), ("cuda:1", "cuda:1"), (" CUDA:01 ", "cuda:1"), ("cuda", "auto"), ("gpu1", "auto")],
)
def test_gpu_device_normalization(raw, expected):
    assert normalize_settings({"gpu_device": raw})["gpu_device"] == expected


@pytest.mark.parametrize(("setting", "device"), [("auto", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:1", "cuda:1")])
def test_runtime_choice_uses_the_selected_card(two_cards, setting, device):
    choice = hardware.resolve_runtime_choice({"execution_target": "nvidia", "gpu_device": setting})
    assert choice.device == device
    assert choice.device_label.startswith(f"GPU {device[-1]}:")


def test_missing_card_falls_back_to_gpu0_with_a_note(two_cards):
    choice = hardware.resolve_runtime_choice({"execution_target": "auto", "gpu_device": "cuda:5"})
    assert choice.device == "cuda:0"
    assert "cuda:5" in choice.detail


def test_cpu_target_ignores_the_gpu(two_cards):
    assert hardware.resolve_runtime_choice({"execution_target": "cpu", "gpu_device": "cuda:1"}).device == "cpu"


def test_options_list_every_card_and_keep_a_missing_saved_one(two_cards):
    options = hardware.gpu_device_options("cuda:3")
    assert [option["value"] for option in options] == ["auto", "cuda:0", "cuda:1", "cuda:3"]
    assert options[2]["label"] == "GPU 1: NVIDIA GeForce RTX 2070 SUPER (8 GB)"
    assert "not detected" in options[3]["label"]
