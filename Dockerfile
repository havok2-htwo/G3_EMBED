# syntax=docker/dockerfile:1.7
# G3_EMBED — local text-embedding service (FastAPI + React SPA).
# CPU / OpenVINO only: no GPU or CUDA runtime is required or installed.

# ---- Stage 1: build the React admin SPA ----
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: python runtime ----
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/.hf

# libgomp1: OpenMP runtime needed by torch/openvino CPU wheels. curl: healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (sentence-transformers dependency). Pulling it from the CPU
# index avoids the multi-GB CUDA build that PyPI's default torch wheel would drag in.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8777/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "backend.genesis_embed_server"]
