from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict

# Number CUDA devices like nvidia-smi / NVML (PCI bus order), so gpu_device "cuda:1" is the card nvidia-smi calls
# GPU 1. CUDA's default order is "fastest first", which differs on mixed boxes (e.g. 3090 + 2070S). Must be set
# before torch initialises CUDA; an explicit environment value wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SETTINGS_FILE = LOGS_DIR / "genesis_embed_settings.json"
LOG_FILE = LOGS_DIR / "embedding_log.jsonl"

HISTORY_MAX_LEN = 200
BATCH_HISTORY_MAX_LEN = 100

DEFAULT_MODEL_ID = "google/embeddinggemma-2"
QWEN3_QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

# max_seq_length: native token window. With it, over-limit texts for a model that is not loaded are rejected
# by its tokenizer alone (no model load) and /v1/models reports max_model_len before loading.
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "google/embeddinggemma-2": {
        "label": "EmbeddingGemma 2",
        "family": "Gemma",
        "tier": "stable",
        "approx_size_gb": 1.5,
        "dimensions": 768,
        "recommended_backend": "cuda",
        "preferred_dtype": "bfloat16",
        "max_seq_length": 8192,
        "default_max_seq_length": 4096,
        "default_encode_batch_size": 4,
        # Short text passes replay captured CUDA graphs (setting cuda_graphs).
        "cuda_graphs": True,
        "query_prompt": "task: search result | query: ",
        "document_prompt": "title: none | text: ",
        "notes": "Default. Multilingual, 8k context, 768 dims. ~1.9 GB VRAM at 4096 tokens, batch 1; ~3.3 GB at batch 4.",
        # Accepts image, audio and video (frame list) inputs; other models are text-only.
        "modalities": ["image", "audio", "video"],
        # Extra VRAM per forward pass, measured bf16 / sdpa / eager on an RTX 4090 (independent of image
        # resolution). Peak ~= max(tower cost, lm_token x rows x longest tokens) + overhead.
        "media_cost_mb": {"image": 74, "video_frame": 36, "audio_second": 3.5, "lm_token": 0.125, "overhead": 50},
    },
    "intfloat/multilingual-e5-small": {
        "label": "E5 Multilingual Small",
        "family": "E5",
        "tier": "stable",
        "approx_size_gb": 0.47,
        "dimensions": 384,
        "recommended_backend": "auto",
        "max_seq_length": 512,
        "query_prompt": "query: ",
        "document_prompt": "passage: ",
        "notes": "Fast multilingual embedding model.",
    },
    "intfloat/multilingual-e5-base": {
        "label": "E5 Multilingual Base",
        "family": "E5",
        "tier": "stable",
        "approx_size_gb": 1.1,
        "dimensions": 768,
        "recommended_backend": "auto",
        "max_seq_length": 512,
        "query_prompt": "query: ",
        "document_prompt": "passage: ",
        "notes": "Balanced multilingual quality and latency.",
    },
    "intfloat/multilingual-e5-large": {
        "label": "E5 Multilingual Large",
        "family": "E5",
        "tier": "stable",
        "approx_size_gb": 2.2,
        "dimensions": 1024,
        "recommended_backend": "cuda",
        "max_seq_length": 512,
        "query_prompt": "query: ",
        "document_prompt": "passage: ",
        "notes": "Higher quality multilingual baseline.",
    },
    "BAAI/bge-m3": {
        "label": "BGE M3",
        "family": "BGE",
        "tier": "stable",
        "approx_size_gb": 2.3,
        "dimensions": 1024,
        "recommended_backend": "cuda",
        "max_seq_length": 8192,
        "notes": "Multilingual dense retrieval model.",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "All MiniLM L6 v2",
        "family": "MiniLM",
        "tier": "diagnostic",
        "approx_size_gb": 0.09,
        "dimensions": 384,
        "recommended_backend": "cpu",
        "max_seq_length": 256,
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
    "Qwen/Qwen3-Embedding-0.6B": {
        "label": "Qwen3 Embedding 0.6B",
        "family": "Qwen3",
        "tier": "stable",
        "approx_size_gb": 1.2,
        "dimensions": 1024,
        "recommended_backend": "cuda",
        "max_seq_length": 32768,
        "preferred_dtype": "bfloat16",
        "quantizable": True,
        "query_prompt": QWEN3_QUERY_PROMPT,
        "default_max_seq_length": 4096,
        "default_encode_batch_size": 2,
        "notes": "Long-context (32k) multilingual model with E5-Large dimensions. ~1.7 GB VRAM at 4096 tokens, batch 1.",
    },
    "Qwen/Qwen3-Embedding-4B": {
        "label": "Qwen3 Embedding 4B",
        "family": "Qwen3",
        "tier": "stable",
        "approx_size_gb": 8.1,
        "dimensions": 2560,
        "recommended_backend": "cuda",
        # Native 40960, capped at MAX_SEQ_LENGTH_LIMIT.
        "max_seq_length": 32768,
        "preferred_dtype": "bfloat16",
        "quantizable": True,
        "query_prompt": QWEN3_QUERY_PROMPT,
        "default_max_seq_length": 4096,
        "default_encode_batch_size": 2,
        "notes": "VRAM at 4096 tokens, batch 1: ~8.7 GB BF16, ~3.8 GB NF4.",
    },
    "Qwen/Qwen3-Embedding-8B": {
        "label": "Qwen3 Embedding 8B",
        "family": "Qwen3",
        "tier": "experimental-xl",
        "approx_size_gb": 15.1,
        "dimensions": 4096,
        "recommended_backend": "cuda",
        # Native 40960, capped at MAX_SEQ_LENGTH_LIMIT.
        "max_seq_length": 32768,
        "preferred_dtype": "bfloat16",
        "quantizable": True,
        "query_prompt": QWEN3_QUERY_PROMPT,
        "default_max_seq_length": 4096,
        "default_encode_batch_size": 2,
        "notes": "VRAM at 4096 tokens, batch 1: ~15.4 GB BF16, ~9 GB INT8, ~5.5-7 GB NF4.",
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

QUANTIZATION_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "INT8 (bitsandbytes)", "value": "int8"},
    {"label": "NF4 4-bit (bitsandbytes)", "value": "nf4"},
]

MAX_SEQ_LENGTH_LIMIT = 32768

# Media input settings: name -> (default, minimum, hard maximum). normalize_settings clamps to these ranges.
MEDIA_LIMIT_SETTINGS: Dict[str, tuple[int, int, int]] = {
    "max_images_per_input": (4, 1, 8),
    "max_images_per_request": (16, 1, 32),
    "max_audio_clips_per_input": (2, 1, 4),
    # The feature extractor silently cuts audio at 30 s; longer clips are rejected instead.
    "max_audio_seconds_per_clip": (30, 1, 30),
    "max_audio_seconds_per_request": (120, 1, 300),
    "max_videos_per_request": (1, 1, 2),
    # 16 frames x (140 + 2) tokens + 2 = 2274 tokens, inside the default 4096 window.
    "max_video_frames": (8, 1, 16),
    "max_media_item_mb": (8, 1, 25),
    "max_image_megapixels": (25, 1, 50),
    "max_request_mb": (32, 1, 256),
    "media_pass_budget_mb": (1200, 200, 4096),
    "max_concurrent_media_requests": (2, 1, 8),
}
MEDIA_TOGGLE_SETTINGS: Dict[str, bool] = {
    "image_inputs_enabled": True,
    "audio_inputs_enabled": True,
    "video_inputs_enabled": True,
}

# Request schema caps (HTTP 422). Bodies are validated before auth, and every bad list item costs one or more
# validation errors, so lists stay short enough that a hostile body cannot fan out into millions of errors.
MAX_INPUTS_PER_REQUEST = 2048
MAX_MEDIA_ITEMS_PER_FIELD = 64
# Body cap (HTTP 413) for every other POST/PUT/PATCH/DELETE route (admin, login); embeddings use max_request_mb.
OTHER_ROUTES_MAX_BODY_MB = 4

TOO_LONG_POLICY_OPTIONS = [
    {"label": "Reject with HTTP 400", "value": "reject"},
    {"label": "Truncate silently", "value": "truncate"},
]

COMPILE_OPTIONS = [
    {"label": "Off", "value": "off"},
    {"label": "Regional (fast warmup)", "value": "regional"},
    {"label": "Full (max throughput, slow warmup)", "value": "full"},
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
