# syntax=docker/dockerfile:1.7
# G3_EMBED — local embedding service (FastAPI + React SPA).
# Target host: Linux + NVIDIA driver + nvidia-container-toolkit. Like the native install it runs EmbeddingGemma 2 in
# bf16 with CUDA graphs on PyTorch CUDA (first start applies the "24GB VRAM" preset); without a GPU the runtime falls
# back to CPU. The image ships NO models: they are downloaded into the mounted /app/models volume from the dashboard.
#
# CPU-only image (no CUDA wheels, much smaller):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t g3-embed:cpu .

# ---- Stage 1: build the React admin SPA ----
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: python runtime ----
FROM python:3.12-slim-bookworm AS runtime

# PyTorch wheel index: CUDA 12.8 by default (same build as the native install, torch 2.11.0+cu128; the wheels bundle
# their own CUDA + cuDNN runtime and Triton, the host driver is injected by nvidia-container-toolkit; cu128 supports
# sm_120). https://download.pytorch.org/whl/cpu gives a CPU-only image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

# OPENBLAS_NUM_THREADS: numpy/scipy's OpenBLAS would start one thread per core, each with a never-used 32 MB buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/.hf \
    TORCHINDUCTOR_CACHE_DIR=/app/models/.inductor \
    TRITON_CACHE_DIR=/app/models/.triton \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OPENBLAS_NUM_THREADS=4
# Smaller caching-allocator reserve: the GPU is shared with other services.
ENV PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

# System libraries:
#   build-essential -> C/C++ toolchain torch.compile / Inductor needs for compile_mode (off by default; without it a
#                      compile request falls back to eager). CUDA graphs need no compiler.
#   libgomp1        -> OpenMP runtime for the torch / OpenVINO CPU paths
#   curl            -> container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch + torchvision first (EmbeddingGemma 2's processor needs torchvision), pinned to the production build. The
# OpenVINO wheel from requirements.txt stays installed for the CPU / Intel fallback targets.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch==2.11.0 torchvision==0.26.0 --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /build/frontend/dist ./frontend/dist
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Non-root runtime; models/logs live in mounted volumes owned by the app user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/logs \
    && chown -R app:app /app
USER app

EXPOSE 8777
VOLUME ["/app/models", "/app/logs"]

# The first start loads, warms up and captures CUDA graphs before /health answers quickly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8777/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "backend.genesis_embed_server"]
