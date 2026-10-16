"""Named settings presets: built-in GPU tiers plus presets the operator saves from the dashboard.

Same model as G3_RERANK and G3_OmniVoice: a preset is a bag of settings values. Loading one fills the settings form;
the operator reviews it and saves through the normal PUT /api/admin/settings, so a preset never bypasses validation.

Built-in tiers (one name per target machine, identical across the G3 services):
- "32GB VRAM"  RTX 5090, every G3 service on one card, whole-card peak <= 30 GB.
- "24GB VRAM"  RTX 4090, every G3 service on one card, whole-card peak <= 23 GB.
- "24+8 VRAM"  3090 (24 GB) + 2070S (8 GB) box, this service on the 24 GB card.
The number in front names the card the service runs on. Presets with a target card size pick gpu_device on load:
the CUDA device whose memory matches (machines number their cards differently), see resolve_presets.
There is no "8+24 VRAM" tier: EMBED runs on the 3090 next to RERANK, since nearly every search uses both.

User presets live in logs/genesis_embed_presets.json (next to the settings file); they cannot reuse a built-in name.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from .genesis_embed_server_globals import LOGS_DIR
from .genesis_embed_server_hardware import get_system_snapshot
from .genesis_embed_server_storage import _atomic_write_text, normalize_settings

PRESETS_FILE = LOGS_DIR / "genesis_embed_presets.json"
PRESET_NAME_MAX_LENGTH = 64

# Keys a preset may set: runtime, VRAM, batching, precision and device behaviour. Machine-specific values
# (model_cache_path), the Hugging Face token, request policy (too_long_policy) and the per-request media limits stay
# out, so loading a preset never touches them.
PRESET_KEYS = (
    "default_model",
    "execution_target",
    "gpu_device",
    "backend_override",
    "precision",
    "quantization",
    "compile_mode",
    "cuda_graphs",
    "warmup_on_load",
    "batching_enabled",
    "batch_wait_time_ms",
    "batch_max_texts",
    "batch_max_chars",
    "encode_batch_size",
    "encode_token_batching",
    "max_seq_length",
    "media_pass_budget_mb",
    "max_concurrent_media_requests",
)

# Tolerance when matching a preset's card size to a detected card (a "24 GB" card reports ~23.6-24.0 GiB).
CARD_MATCH_TOLERANCE_MIB = 2048

# ---------------------------------------------------------------------------------------------------------------
# BUILT-IN TIER VALUES. Placeholders: every tier currently carries the service defaults (DEFAULT_SETTINGS) for its
# keys. They are replaced by values measured on each target machine in a later step; do not tune them by guess.
# gpu_device is never listed here: single-card tiers leave it alone, two-card tiers get it from card matching.
# ---------------------------------------------------------------------------------------------------------------
BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "32GB VRAM": {
        "description": "RTX 5090, all G3 services on one card (whole-card peak <= 30 GB). Values not measured yet: "
        "currently the service defaults.",
        "target_vram_gb": None,
        "values": {
            "default_model": "google/embeddinggemma-2",
            "execution_target": "auto",
            "backend_override": "auto",
            "precision": "auto",
            "quantization": "none",
            "compile_mode": "off",
            "cuda_graphs": True,
            "warmup_on_load": False,
            "batching_enabled": True,
            "batch_wait_time_ms": 25,
            "batch_max_texts": 32,
            "batch_max_chars": 64000,
            "encode_batch_size": 0,
            "encode_token_batching": True,
            "max_seq_length": 0,
            "media_pass_budget_mb": 1200,
            "max_concurrent_media_requests": 2,
        },
    },
    "24GB VRAM": {
        "description": "RTX 4090, all G3 services on one card (whole-card peak <= 23 GB). Values not measured yet: "
        "currently the service defaults.",
        "target_vram_gb": None,
        "values": {
            "default_model": "google/embeddinggemma-2",
            "execution_target": "auto",
            "backend_override": "auto",
            "precision": "auto",
            "quantization": "none",
            "compile_mode": "off",
            "cuda_graphs": True,
            "warmup_on_load": False,
            "batching_enabled": True,
            "batch_wait_time_ms": 25,
            "batch_max_texts": 32,
            "batch_max_chars": 64000,
            "encode_batch_size": 0,
            "encode_token_batching": True,
            "max_seq_length": 0,
            "media_pass_budget_mb": 1200,
            "max_concurrent_media_requests": 2,
        },
    },
    "24+8 VRAM": {
        "description": "3090 + 2070S box, EMBED on the 3090 (24 GB) next to RERANK. Values not measured yet: "
        "currently the service defaults.",
        "target_vram_gb": 24,
        "values": {
            "default_model": "google/embeddinggemma-2",
            "execution_target": "auto",
            "backend_override": "auto",
            "precision": "auto",
            "quantization": "none",
            "compile_mode": "off",
            "cuda_graphs": True,
            "warmup_on_load": False,
            "batching_enabled": True,
            "batch_wait_time_ms": 25,
            "batch_max_texts": 32,
            "batch_max_chars": 64000,
            "encode_batch_size": 0,
            "encode_token_batching": True,
            "max_seq_length": 0,
            "media_pass_budget_mb": 1200,
            "max_concurrent_media_requests": 2,
        },
    },
}

_lock = threading.RLock()


def _clean_name(name: str) -> str:
    clean = " ".join(str(name or "").split())
    if not clean:
        raise ValueError("Preset name is required.")
    if len(clean) > PRESET_NAME_MAX_LENGTH:
        raise ValueError(f"Preset names have at most {PRESET_NAME_MAX_LENGTH} characters.")
    return clean


def preset_values(values: Dict[str, Any]) -> Dict[str, Any]:
    """Only PRESET_KEYS, each normalised exactly like a saved setting (unknown keys are dropped)."""
    sent = {key: values[key] for key in PRESET_KEYS if key in values}
    normalized = normalize_settings(sent)
    return {key: normalized[key] for key in sent}


def _load_user_presets() -> Dict[str, Dict[str, Any]]:
    if not PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[G3_EMBED] Could not load presets: {exc}", file=sys.stderr, flush=True)
        return {}
    presets: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get("values"), dict) and name not in BUILTIN_PRESETS:
                presets[str(name)] = {
                    "name": str(name),
                    "values": preset_values(entry["values"]),
                    "created_at": entry.get("created_at"),
                }
    return presets


def _match_card(target_gb: Optional[int], cards_mib: List[int]) -> tuple[Optional[str], Optional[str]]:
    """gpu_device for a preset's card size on this machine, plus a note when no card matches."""
    if not target_gb:
        return None, None
    target_mib = target_gb * 1024
    for index, total_mib in enumerate(cards_mib):
        if abs(total_mib - target_mib) <= CARD_MATCH_TOLERANCE_MIB:
            return f"cuda:{index}", None
    detected = ", ".join(f"GPU {index}: {total / 1024:.0f} GB" for index, total in enumerate(cards_mib)) or "none"
    return None, f"No {target_gb} GB card detected (cards: {detected}); the GPU setting is left unchanged."


def resolve_presets() -> List[Dict[str, Any]]:
    """Built-ins first (in tier order), then user presets by name. Built-ins with a card size get gpu_device."""
    cards_mib = list(get_system_snapshot()["torch"].get("cuda_memory_mib") or [])
    result: List[Dict[str, Any]] = []
    for name, preset in BUILTIN_PRESETS.items():
        values = dict(preset["values"])
        device, note = _match_card(preset.get("target_vram_gb"), cards_mib)
        if device:
            values["gpu_device"] = device
        result.append(
            {
                "name": name,
                "built_in": True,
                "description": preset.get("description", ""),
                "target_vram_gb": preset.get("target_vram_gb"),
                "note": note,
                "values": values,
            }
        )
    with _lock:
        user = _load_user_presets()
    for name in sorted(user, key=str.lower):
        entry = user[name]
        result.append(
            {
                "name": name,
                "built_in": False,
                "description": "",
                "created_at": entry.get("created_at"),
                "note": None,
                "values": entry["values"],
            }
        )
    return result


def save_user_preset(name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    clean = _clean_name(name)
    if clean in BUILTIN_PRESETS:
        raise ValueError(f"'{clean}' is a built-in preset; save under another name.")
    cleaned = preset_values(values)
    if not cleaned:
        raise ValueError("The preset contains no settings.")
    with _lock:
        presets = _load_user_presets()
        created = (presets.get(clean) or {}).get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        presets[clean] = {"name": clean, "values": cleaned, "created_at": created}
        _atomic_write_text(PRESETS_FILE, json.dumps(presets, indent=2, ensure_ascii=True))
    return {"name": clean, "built_in": False, "description": "", "created_at": created, "note": None, "values": cleaned}


def delete_user_preset(name: str) -> bool:
    clean = " ".join(str(name or "").split())
    if clean in BUILTIN_PRESETS:
        raise ValueError(f"'{clean}' is a built-in preset and cannot be deleted.")
    with _lock:
        presets = _load_user_presets()
        if clean not in presets:
            return False
        presets.pop(clean)
        _atomic_write_text(PRESETS_FILE, json.dumps(presets, indent=2, ensure_ascii=True))
    return True
