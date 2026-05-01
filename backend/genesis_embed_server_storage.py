from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from .genesis_embed_server_globals import (
    BACKEND_OPTIONS,
    COMPILE_OPTIONS,
    EXECUTION_TARGET_OPTIONS,
    LOG_FILE,
    LOGS_DIR,
    MODEL_SPECS,
    PRECISION_OPTIONS,
    SETTINGS_FILE,
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "default_model": "intfloat/multilingual-e5-small",
    "execution_target": "auto",
    "backend_override": "auto",
    "model_cache_path": ".\\models",
    "batching_enabled": True,
    "batch_wait_time_ms": 25,
    "batch_max_texts": 32,
    "batch_max_chars": 64000,
    "precision": "auto",
    "compile_mode": "off",
    "warmup_on_load": False,
    "huggingface_token": "",
}


def _valid_option(options: list[dict[str, str]], value: Any, default: str) -> str:
    normalized = str(value or default).strip()
    valid_values = {option["value"] for option in options}
    return normalized if normalized in valid_values else default


def _clamp_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = settings or {}
    default_model = str(source.get("default_model", DEFAULT_SETTINGS["default_model"])).strip()
    if default_model not in MODEL_SPECS:
        default_model = DEFAULT_SETTINGS["default_model"]

    return {
        "default_model": default_model,
        "execution_target": _valid_option(EXECUTION_TARGET_OPTIONS, source.get("execution_target"), "auto"),
        "backend_override": _valid_option(BACKEND_OPTIONS, source.get("backend_override"), "auto"),
        "model_cache_path": str(source.get("model_cache_path", DEFAULT_SETTINGS["model_cache_path"])).strip(),
        "batching_enabled": bool(source.get("batching_enabled", DEFAULT_SETTINGS["batching_enabled"])),
        "batch_wait_time_ms": _clamp_int(
            source.get("batch_wait_time_ms", DEFAULT_SETTINGS["batch_wait_time_ms"]),
            minimum=0,
            maximum=5000,
            default=DEFAULT_SETTINGS["batch_wait_time_ms"],
        ),
        "batch_max_texts": _clamp_int(
            source.get("batch_max_texts", DEFAULT_SETTINGS["batch_max_texts"]),
            minimum=1,
            maximum=2048,
            default=DEFAULT_SETTINGS["batch_max_texts"],
        ),
        "batch_max_chars": _clamp_int(
            source.get("batch_max_chars", DEFAULT_SETTINGS["batch_max_chars"]),
            minimum=256,
            maximum=4_000_000,
            default=DEFAULT_SETTINGS["batch_max_chars"],
        ),
        "precision": _valid_option(PRECISION_OPTIONS, source.get("precision"), "auto"),
        "compile_mode": _valid_option(COMPILE_OPTIONS, source.get("compile_mode"), "off"),
        "warmup_on_load": bool(source.get("warmup_on_load", DEFAULT_SETTINGS["warmup_on_load"])),
        "huggingface_token": str(source.get("huggingface_token", "")).strip(),
    }


def load_settings() -> Dict[str, Any]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        return normalize_settings(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[G3_EMBED] Could not load settings: {exc}", file=sys.stderr)
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_settings(settings)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        SETTINGS_FILE.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
    except OSError as exc:
        print(f"[G3_EMBED] Could not save settings: {exc}", file=sys.stderr)
    return normalized


def log_embedding(entry: Dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError as exc:
        print(f"[G3_EMBED] Could not write embedding log: {exc}", file=sys.stderr)
