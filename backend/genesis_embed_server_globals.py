from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SETTINGS_FILE = LOGS_DIR / "genesis_embed_settings.json"
LOG_FILE = LOGS_DIR / "embedding_log.jsonl"

HISTORY_MAX_LEN = 200
BATCH_HISTORY_MAX_LEN = 100

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "intfloat/multilingual-e5-small": {
        "label": "E5 Multilingual Small",
        "family": "E5",
        "tier": "stable",
        "approx_size_gb": 0.47,
        "dimensions": 384,
        "recommended_backend": "auto",
        "notes": "Fast default multilingual embedding model.",
    },
    "intfloat/multilingual-e5-base": {
        "label": "E5 Multilingual Base",
        "family": "E5",
        "tier": "stable",
        "approx_size_gb": 1.1,
        "dimensions": 768,
        "recommended_backend": "auto",
        "notes": "Balanced multilingual quality and latency.",
    },
    "intfloat/multilingual-e5-large": {
        "label": "E5 Multilingual Large",
        "family": "E5",
        "tier": "stable",
        "approx_size_gb": 2.2,
        "dimensions": 1024,
        "recommended_backend": "cuda",
        "notes": "Higher quality multilingual baseline.",
    },
    "BAAI/bge-m3": {
        "label": "BGE M3",
        "family": "BGE",
        "tier": "stable",
        "approx_size_gb": 2.3,
        "dimensions": 1024,
        "recommended_backend": "cuda",
        "notes": "Multilingual dense retrieval model.",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "All MiniLM L6 v2",
        "family": "MiniLM",
        "tier": "diagnostic",
        "approx_size_gb": 0.09,
        "dimensions": 384,
        "recommended_backend": "cpu",
        "notes": "Tiny English diagnostic model for quick checks.",
    },
    "BAAI/bge-multilingual-gemma2": {
        "label": "BGE Multilingual Gemma2",
        "family": "BGE",
        "tier": "experimental-xl",
        "approx_size_gb": 18.0,
        "dimensions": None,
        "recommended_backend": "cuda",
        "notes": "Large experimental model for high-VRAM tests.",
    },
}

MODEL_OPTIONS = [{"label": spec["label"], "value": model_id} for model_id, spec in MODEL_SPECS.items()]

EXECUTION_TARGET_OPTIONS = [
    {"label": "Auto", "value": "auto"},
    {"label": "NVIDIA RTX / CUDA", "value": "nvidia"},
    {"label": "Intel NPU", "value": "intel_npu"},
    {"label": "Intel iGPU", "value": "intel_igpu"},
    {"label": "CPU", "value": "cpu"},
]

BACKEND_OPTIONS = [
    {"label": "Auto", "value": "auto"},
    {"label": "PyTorch", "value": "pytorch"},
    {"label": "OpenVINO", "value": "openvino"},
]

PRECISION_OPTIONS = [
    {"label": "Auto", "value": "auto"},
    {"label": "Float32", "value": "float32"},
    {"label": "Float16", "value": "float16"},
    {"label": "BFloat16", "value": "bfloat16"},
]

COMPILE_OPTIONS = [
    {"label": "Off", "value": "off"},
    {"label": "Reduce overhead", "value": "reduce-overhead"},
]

settings_lock = threading.RLock()
history_lock = threading.RLock()
batch_state_lock = threading.RLock()

current_settings: Dict[str, Any] = {}
embedding_history = deque(maxlen=HISTORY_MAX_LEN)
batch_history = deque(maxlen=BATCH_HISTORY_MAX_LEN)

batch_runtime_state: Dict[str, Any] = {
    "worker_running": False,
    "queue_size": 0,
    "active_batch_id": None,
    "active_batch_size": 0,
    "active_batch_chars": 0,
    "active_batch_started_at": None,
    "last_batch_completed_at": None,
    "last_batch_duration_ms": None,
    "last_batch_texts_per_second": None,
    "last_error": None,
    "total_batches_processed": 0,
    "total_texts_processed": 0,
}


def resolve_model_cache_path(cache_path: str) -> str:
    normalized = str(cache_path or "").strip()
    if not normalized:
        return ""
    path = Path(normalized).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve(strict=False))


def get_model_spec(model_id: str) -> Dict[str, Any]:
    return MODEL_SPECS.get(model_id) or {
        "label": model_id,
        "family": "custom",
        "tier": "custom",
        "approx_size_gb": None,
        "dimensions": None,
        "recommended_backend": "auto",
        "notes": "Custom model id.",
    }
