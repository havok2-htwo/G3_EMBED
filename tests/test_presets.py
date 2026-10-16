"""Settings presets: built-in GPU tiers, card matching for the two-GPU tier, saved presets and the admin routes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import genesis_embed_server_presets as presets
from backend.genesis_embed_server_storage import normalize_settings

TIER_NAMES = ["32GB VRAM", "24GB VRAM", "24+8 VRAM"]


def _cards(monkeypatch, *cards_mib: int) -> None:
    snapshot = {"torch": {"cuda_devices": [f"GPU{index}" for index in range(len(cards_mib))], "cuda_memory_mib": list(cards_mib)}}
    monkeypatch.setattr(presets, "get_system_snapshot", lambda: snapshot)


def test_builtin_tiers_come_first_in_tier_order(monkeypatch):
    _cards(monkeypatch, 32607)
    names = [preset["name"] for preset in presets.resolve_presets()]
    assert names[: len(TIER_NAMES)] == TIER_NAMES
    assert list(presets.BUILTIN_PRESETS) == TIER_NAMES


@pytest.mark.parametrize("name", TIER_NAMES)
def test_builtin_values_are_valid_settings(name):
    values = presets.BUILTIN_PRESETS[name]["values"]
    assert set(values) <= set(presets.PRESET_KEYS)
    assert "gpu_device" not in values, "the GPU comes from card matching, never from the tier values"
    normalized = normalize_settings(values)
    assert {key: normalized[key] for key in values} == values, "a built-in value would be clamped or rejected"


def test_builtin_values_are_the_measured_tier_values():
    # Measured on an RTX 5090 (see the BUILT-IN TIER VALUES comment): only the GPU batch size differs per tier.
    common = {
        "default_model": "google/embeddinggemma-2",
        "precision": "auto",
        "quantization": "none",
        "compile_mode": "off",
        "cuda_graphs": True,
        "encode_token_batching": True,
        "max_seq_length": 0,
        "idle_trim_seconds": 5,
    }
    batch_sizes = {"32GB VRAM": 4, "24GB VRAM": 2, "24+8 VRAM": 2}
    for name, batch_size in batch_sizes.items():
        values = presets.BUILTIN_PRESETS[name]["values"]
        assert {key: values[key] for key in common} == common, name
        assert values["encode_batch_size"] == batch_size, name
        assert set(values) == set(presets.PRESET_KEYS) - {"gpu_device"}, f"{name} leaves a preset key to chance"
    assert presets.BUILTIN_PRESETS["24+8 VRAM"]["target_vram_gb"] == 24
    assert "8+24 VRAM" not in presets.BUILTIN_PRESETS


@pytest.mark.parametrize(("name", "idle_gb", "peak_gb"), [("32GB VRAM", "3.0", "5.3"), ("24GB VRAM", "3.0", "4.1"), ("24+8 VRAM", "3.0", "4.1")])
def test_builtin_descriptions_state_the_measured_vram(name, idle_gb, peak_gb):
    description = presets.BUILTIN_PRESETS[name]["description"]
    assert f"{idle_gb} GB idle" in description and f"{peak_gb} GB peak" in description
    assert "measured on the RTX 5090" in description
    assert "not measured" not in description


def test_preset_keys_hold_no_secrets_or_paths():
    for key in ("huggingface_token", "model_cache_path"):
        assert key not in presets.PRESET_KEYS
    assert set(presets.PRESET_KEYS) <= set(normalize_settings({}))


@pytest.mark.parametrize(("cards", "expected"), [((24576, 8192), "cuda:0"), ((8192, 24576), "cuda:1"), ((8192, 24135), "cuda:1")])
def test_two_gpu_tier_picks_the_24gb_card_wherever_it_sits(monkeypatch, cards, expected):
    _cards(monkeypatch, *cards)
    tier = next(preset for preset in presets.resolve_presets() if preset["name"] == "24+8 VRAM")
    assert tier["values"]["gpu_device"] == expected
    assert tier["note"] is None


def test_two_gpu_tier_without_a_matching_card_leaves_the_gpu_alone(monkeypatch):
    _cards(monkeypatch, 32607)
    tier = next(preset for preset in presets.resolve_presets() if preset["name"] == "24+8 VRAM")
    assert "gpu_device" not in tier["values"]
    assert "No 24 GB card" in tier["note"]


def test_two_gpu_tier_without_any_cuda_card(monkeypatch):
    _cards(monkeypatch)
    tier = next(preset for preset in presets.resolve_presets() if preset["name"] == "24+8 VRAM")
    assert "gpu_device" not in tier["values"]
    assert "cards: none" in tier["note"]


def test_single_card_tiers_never_set_the_gpu(monkeypatch):
    _cards(monkeypatch, 24576, 8192)
    for preset in presets.resolve_presets():
        if preset["name"] in ("32GB VRAM", "24GB VRAM"):
            assert "gpu_device" not in preset["values"]
            assert preset["note"] is None


def test_saved_preset_keeps_only_preset_keys_normalised(monkeypatch, tmp_path):
    _cards(monkeypatch)
    saved = presets.save_user_preset(
        "  my   tier ",
        {
            "batch_max_texts": 999999,
            "huggingface_token": "hf_secret",
            "model_cache_path": "D:\\models",
            "too_long_policy": "truncate",
            "nonsense": 1,
            "precision": "bfloat16",
            "gpu_device": " CUDA:01 ",
        },
    )
    assert saved["name"] == "my tier"
    assert saved["values"] == {"batch_max_texts": 2048, "precision": "bfloat16", "gpu_device": "cuda:1"}
    listed = [preset for preset in presets.resolve_presets() if not preset["built_in"]]
    assert [(preset["name"], preset["values"]) for preset in listed] == [("my tier", saved["values"])]
    assert presets.PRESETS_FILE == tmp_path / "genesis_embed_presets.json"
    stored = presets.PRESETS_FILE.read_text(encoding="utf-8")
    assert "hf_secret" not in stored and "D:" not in stored
    assert json.loads(stored)["my tier"]["values"] == saved["values"]
    assert not list(tmp_path.glob("*.tmp")), "the atomic write left its temp file behind"


def test_saving_again_keeps_the_creation_time(monkeypatch):
    _cards(monkeypatch)
    first = presets.save_user_preset("lab", {"batch_max_texts": 16})
    second = presets.save_user_preset("lab", {"batch_max_texts": 64})
    assert second["created_at"] == first["created_at"]
    assert [preset["values"] for preset in presets.resolve_presets() if preset["name"] == "lab"] == [{"batch_max_texts": 64}]


def test_corrupt_presets_file_is_ignored(monkeypatch):
    _cards(monkeypatch)
    presets.PRESETS_FILE.write_text("{not json", encoding="utf-8")
    assert [preset["name"] for preset in presets.resolve_presets()] == TIER_NAMES


def test_builtin_names_are_protected():
    with pytest.raises(ValueError):
        presets.save_user_preset("32GB VRAM", {"batch_max_texts": 16})
    with pytest.raises(ValueError):
        presets.save_user_preset(" 24+8   VRAM ", {"batch_max_texts": 16})
    with pytest.raises(ValueError):
        presets.delete_user_preset("24GB VRAM")


def test_delete_saved_preset(monkeypatch):
    _cards(monkeypatch)
    presets.save_user_preset("tmp", {"batch_max_texts": 16})
    assert presets.delete_user_preset("tmp") is True
    assert presets.delete_user_preset("tmp") is False
    assert all(preset["built_in"] for preset in presets.resolve_presets())


def test_empty_or_long_names_are_rejected():
    with pytest.raises(ValueError):
        presets.save_user_preset("   ", {"batch_max_texts": 16})
    with pytest.raises(ValueError):
        presets.save_user_preset("x" * 65, {"batch_max_texts": 16})
    with pytest.raises(ValueError):
        presets.save_user_preset("only unknown keys", {"nonsense": 1, "huggingface_token": "hf_x"})


def test_admin_preset_routes(admin_client, monkeypatch):
    _cards(monkeypatch, 32607)
    listed = admin_client.get("/api/admin/settings/presets").json()["presets"]
    assert [preset["name"] for preset in listed] == TIER_NAMES

    response = admin_client.put("/api/admin/settings/presets/lab", json={"values": {"batch_max_texts": 16, "gpu_device": "cuda:1"}})
    assert response.status_code == 200, response.text
    assert response.json()["preset"]["values"] == {"batch_max_texts": 16, "gpu_device": "cuda:1"}
    assert [preset["name"] for preset in admin_client.get("/api/admin/settings/presets").json()["presets"]] == [*TIER_NAMES, "lab"]

    assert admin_client.put("/api/admin/settings/presets/32GB VRAM", json={"values": {"batch_max_texts": 16}}).status_code == 400
    assert admin_client.put("/api/admin/settings/presets/empty", json={"values": {}}).status_code == 400
    assert admin_client.delete("/api/admin/settings/presets/24GB VRAM").status_code == 400
    assert admin_client.delete("/api/admin/settings/presets/lab").status_code == 200
    assert admin_client.delete("/api/admin/settings/presets/lab").status_code == 404


def test_first_start_applies_the_24gb_tier_and_saves_it():
    from backend import genesis_embed_server_storage as storage

    assert not storage.SETTINGS_FILE.exists()
    settings, applied = presets.load_settings_or_first_start_preset()
    assert applied is True
    for key, value in presets.BUILTIN_PRESETS["24GB VRAM"]["values"].items():
        assert settings[key] == value, key
    assert settings["gpu_device"] == "auto", "a single-card tier never sets the GPU"
    assert storage.SETTINGS_FILE.exists()
    assert storage.load_settings() == settings


def test_saved_settings_are_kept_on_later_starts():
    from backend import genesis_embed_server_storage as storage

    storage.save_settings({**storage.DEFAULT_SETTINGS, "encode_batch_size": 8, "cuda_graphs": False})
    settings, applied = presets.load_settings_or_first_start_preset()
    assert applied is False
    assert (settings["encode_batch_size"], settings["cuda_graphs"]) == (8, False)


def test_docker_entrypoint_seed_still_counts_as_first_start():
    from backend import genesis_embed_server_storage as storage

    script = (Path(__file__).resolve().parent.parent / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    seed = script.split("<<'JSON'\n", 1)[1].split("\nJSON\n", 1)[0]
    storage.SETTINGS_FILE.write_text(seed, encoding="utf-8")
    settings, applied = presets.load_settings_or_first_start_preset()
    assert applied is True
    assert settings["model_cache_path"] == "/app/models"
    assert settings["encode_batch_size"] == presets.BUILTIN_PRESETS["24GB VRAM"]["values"]["encode_batch_size"]
    saved = json.loads(storage.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert presets.FIRST_START_SEED_MARKER not in saved
    assert presets.load_settings_or_first_start_preset()[1] is False
