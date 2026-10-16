"""Named settings presets: built-in GPU tiers plus presets the operator saves from the dashboard.

Same model as G3_RERANK and G3_OmniVoice: a preset is a bag of settings values. Loading one fills the settings form;
the operator reviews it and saves through the normal PUT /api/admin/settings, so a preset never bypasses validation.

Built-in tiers (one name per target machine, identical across the G3 services):
- "32GB VRAM"  RTX 5090, every G3 service on one card, whole-card peak <= 30 GB.
- "24GB VRAM"  RTX 4090, every G3 service on one card, whole-card peak <= 23 GB.
- "24+8 VRAM"  3090 (24 GB) + 2070S (8 GB) box, this service on the 24 GB card.
The number in front names the card the service runs on. Presets with a target card size pick gpu_device on load:
the CUDA device whose memory matches (machines number their cards differently), see resolve_presets.
There is no "8+24 VRAM" tier: EMBED runs on the 3090 next to RERANK, since nearly every search uses both, and the
2070S (Turing) has no bf16 for EmbeddingGemma 2.

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
    "idle_trim_seconds",
)

# Tolerance when matching a preset's card size to a detected card (a "24 GB" card reports ~23.6-24.0 GiB).
CARD_MATCH_TOLERANCE_MIB = 2048

# ---------------------------------------------------------------------------------------------------------------
# BUILT-IN TIER VALUES, measured on an RTX 5090 (2026-10-16, EmbeddingGemma 2 bf16, process VRAM from nvidia-smi
# minus the desktop baseline). The token budget per pass (encode_batch_size x 4096) sizes the text peak whatever the
# request holds: ebs 1/2/4/8/16 peak at 3.5/4.1/5.3/7.6/12.3 GB, and the 5090 is fastest at 4 (512-token chunks
# 291/298/327/305/266 per s, 4096-token texts 60/72/79/77/73k tok/s), so 8 and 16 cost VRAM for less throughput.
# ebs 2 keeps ~91 % of the ebs-4 ingestion throughput for 1.2 GB less. CUDA graphs: 5.5 vs 27.5 ms per query for
# ~0.95 GB resident. compile stays off: on this Windows box Triton cannot build its helper and dynamo ran eager.
# max_seq_length 0 = the 4096 default window (8192 would peak at 7.6 GB at ebs 4). The idle trim hands the pass
# working set back 5 s after the last request, so between requests the process is back at ~3.0 GB.
# The other keys keep the service defaults (media peaks stay below the text peak at the 1200 MB budget).
# gpu_device is never listed here: single-card tiers leave it alone, two-card tiers get it from card matching.
# There is no "8+24 VRAM" tier: the 2070S (Turing) has no bf16, and "auto" would run the bf16-trained Gemma in fp16.
# ---------------------------------------------------------------------------------------------------------------
_TIER_COMMON: Dict[str, Any] = {
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
    "encode_batch_size": 4,
    "encode_token_batching": True,
    "max_seq_length": 0,
    "media_pass_budget_mb": 1200,
    "max_concurrent_media_requests": 2,
    "idle_trim_seconds": 5,
}
BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "32GB VRAM": {
        "description": "RTX 5090, all G3 services on one card. EmbeddingGemma 2 bf16, GPU batch 4 (16384 tokens per "
        "pass), CUDA graphs: about 3.0 GB idle, 5.3 GB peak for this process (measured on the RTX 5090).",
        "target_vram_gb": None,
        "values": dict(_TIER_COMMON),
    },
    "24GB VRAM": {
        "description": "RTX 4090, all G3 services on one card. EmbeddingGemma 2 bf16, GPU batch 2 (8192 tokens per "
        "pass, ~91 % of the batch-4 ingestion speed), CUDA graphs: about 3.0 GB idle, 4.1 GB peak for this process "
        "(measured on the RTX 5090).",
        "target_vram_gb": None,
        "values": {**_TIER_COMMON, "encode_batch_size": 2},
    },
    "24+8 VRAM": {
        "description": "3090 + 2070S box, EMBED on the 3090 (24 GB) next to RERANK. EmbeddingGemma 2 bf16 (Ampere "
        "has bf16), GPU batch 2, CUDA graphs: about 3.0 GB idle, 4.1 GB peak for this process (measured on the "
        "RTX 5090; throughput on the 3090 is lower).",
        "target_vram_gb": 24,
        "values": {**_TIER_COMMON, "encode_batch_size": 2},
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
