"""Media settings: defaults, clamping and admin payload defaults."""
from __future__ import annotations

import json

import pytest

from backend.genesis_embed_server_admin import AdminSettingsPayload
from backend.genesis_embed_server_globals import MEDIA_LIMIT_SETTINGS, MEDIA_TOGGLE_SETTINGS, current_settings
from backend.genesis_embed_server_storage import DEFAULT_SETTINGS, normalize_settings, save_settings, load_settings


def test_defaults_match_spec_table():
    expected = {
        "max_images_per_input": (4, 1, 8),
        "max_images_per_request": (16, 1, 32),
        "max_audio_clips_per_input": (2, 1, 4),
        "max_audio_seconds_per_clip": (30, 1, 30),
        "max_audio_seconds_per_request": (120, 1, 300),
        "max_videos_per_request": (1, 1, 2),
        "max_video_frames": (8, 1, 16),
        "max_media_item_mb": (8, 1, 25),
        "max_image_megapixels": (25, 1, 50),
        "max_request_mb": (32, 1, 256),
        "media_pass_budget_mb": (1200, 200, 4096),
        "max_concurrent_media_requests": (2, 1, 8),
    }
    assert MEDIA_LIMIT_SETTINGS == expected
    for name, (default, _, _) in expected.items():
        assert DEFAULT_SETTINGS[name] == default
    for name in ("image_inputs_enabled", "audio_inputs_enabled", "video_inputs_enabled"):
        assert DEFAULT_SETTINGS[name] is True


@pytest.mark.parametrize("name", sorted(MEDIA_LIMIT_SETTINGS))
def test_normalize_clamps_every_limit(name):
    default, minimum, maximum = MEDIA_LIMIT_SETTINGS[name]
    assert normalize_settings({name: maximum + 1000})[name] == maximum
    assert normalize_settings({name: minimum - 1000})[name] == minimum
    assert normalize_settings({name: "not a number"})[name] == max(minimum, min(maximum, default))
    assert normalize_settings({})[name] == default
    assert normalize_settings({name: minimum})[name] == minimum


@pytest.mark.parametrize("name", sorted(MEDIA_TOGGLE_SETTINGS))
def test_normalize_toggles(name):
    assert normalize_settings({})[name] is True
    assert normalize_settings({name: False})[name] is False
    assert normalize_settings({name: 0})[name] is False


def test_admin_payload_defaults_match_storage_defaults():
    payload = AdminSettingsPayload(
        default_model="google/embeddinggemma-2",
        execution_target="auto",
        backend_override="auto",
        model_cache_path=".\\models",
        batching_enabled=True,
        batch_wait_time_ms=25,
        batch_max_texts=32,
        batch_max_chars=64000,
        precision="auto",
        compile_mode="off",
        warmup_on_load=False,
        huggingface_token="",
    )
    data = payload.model_dump()
    for name in [*MEDIA_LIMIT_SETTINGS, *MEDIA_TOGGLE_SETTINGS]:
        assert data[name] == DEFAULT_SETTINGS[name], name
    assert normalize_settings(data) == normalize_settings(DEFAULT_SETTINGS)


def test_settings_round_trip_uses_isolated_file(tmp_path):
    saved = save_settings({**DEFAULT_SETTINGS, "max_video_frames": 99, "video_inputs_enabled": False})
    assert saved["max_video_frames"] == 16 and saved["video_inputs_enabled"] is False
    assert (tmp_path / "genesis_embed_settings.json").exists()
    assert load_settings()["max_video_frames"] == 16


REQUIRED_FIELDS = {
    "default_model": "google/embeddinggemma-2",
    "execution_target": "auto",
    "backend_override": "auto",
    "model_cache_path": "./models",
    "batching_enabled": True,
    "batch_wait_time_ms": 25,
    "batch_max_texts": 32,
    "batch_max_chars": 64000,
    "precision": "auto",
    "compile_mode": "off",
    "warmup_on_load": False,
    "huggingface_token": "",
}


def test_put_settings_keeps_omitted_fields(admin_client, settings, tmp_path):
    # An older client (or script) that does not know the media fields must not reset them to their defaults.
    settings(video_inputs_enabled=False, media_pass_budget_mb=600, max_seq_length=2048, too_long_policy="truncate")
    response = admin_client.put("/api/admin/settings", json={**REQUIRED_FIELDS, "batch_max_texts": 64})
    assert response.status_code == 200, response.text
    for saved in (response.json()["settings"], current_settings, json.loads((tmp_path / "genesis_embed_settings.json").read_text())):
        assert saved["batch_max_texts"] == 64
        assert saved["video_inputs_enabled"] is False and saved["media_pass_budget_mb"] == 600
        assert saved["max_seq_length"] == 2048 and saved["too_long_policy"] == "truncate"


def test_put_settings_applies_sent_fields(admin_client, settings):
    settings(video_inputs_enabled=False)
    response = admin_client.put(
        "/api/admin/settings", json={**REQUIRED_FIELDS, "video_inputs_enabled": True, "max_video_frames": 99}
    )
    assert response.status_code == 200
    assert current_settings["video_inputs_enabled"] is True and current_settings["max_video_frames"] == 16


def test_put_settings_idle_trim_seconds(admin_client, settings):
    settings(idle_trim_seconds=30)
    assert admin_client.put("/api/admin/settings", json=REQUIRED_FIELDS).status_code == 200
    assert current_settings["idle_trim_seconds"] == 30, "an omitted idle_trim_seconds keeps its value"
    assert admin_client.put("/api/admin/settings", json={**REQUIRED_FIELDS, "idle_trim_seconds": 99999}).status_code == 200
    assert current_settings["idle_trim_seconds"] == 3600
