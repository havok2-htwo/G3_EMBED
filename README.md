# G3_EMBED

`G3_EMBED` is a local FastAPI + React/Vite embedding server extracted from the Genesis3 memory-vector path and shaped like `G3_WHISPER`.

It serves the public embedding API, OpenAPI docs, landing page, and protected admin dashboard from one process on port `8777`.

## Features

- Genesis-compatible `POST /embed`
- OpenAI-compatible `POST /v1/embeddings` and `GET /v1/models`
- `GET /health` with runtime and hardware status
- Admin dashboard at `/admin`, protected by username/password login
- Admin-managed client API keys for public endpoints
- Model dropdown and cache manager
- Dynamic batching by `(model_id, backend, target, precision)`
- Hybrid runtime selection:
  - `auto`: PyTorch CUDA -> OpenVINO NPU -> OpenVINO GPU -> PyTorch XPU -> CPU
  - `nvidia`: PyTorch CUDA
  - `intel_npu`: OpenVINO NPU
  - `intel_igpu`: OpenVINO GPU, then PyTorch XPU fallback
  - `cpu`: CPU fallback
- Embedding Lab with two-text vector generation, cosine similarity, dot product, distance, timings, and raw vector copy

## Models

The v1 model catalog is intentionally curated:

- `intfloat/multilingual-e5-small` default
- `intfloat/multilingual-e5-base`
- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-multilingual-gemma2` experimental XL option for high-VRAM tests

Embeddings are normalized by default and no automatic `query:` / `passage:` prefixes are added.

## Quick Start

Windows:

```bat
start.bat
```

Linux/macOS:

```bash
bash ./start.sh
```

The scripts create `venv`, install a suitable PyTorch backend, install Python and frontend dependencies, build the frontend, and start the server.

Default URLs:

- Landing page: `http://127.0.0.1:8777/`
- Admin dashboard: `http://127.0.0.1:8777/admin`
- OpenAPI docs: `http://127.0.0.1:8777/docs`
- Native API: `POST http://127.0.0.1:8777/embed`

## Admin Login and Client API Keys

Admin users log in at `/admin` with a username and password. The default first-run login is `admin` / `admin`, and the server requires changing that password before protected admin actions are available.

Client API keys are managed from the admin dashboard. Public embedding endpoints remain open until at least one client key exists; once keys exist, callers must send `X-API-Key`.

Optional environment variables:

- `GENESIS_TORCH_VARIANT=auto|cpu|cuda|xpu`
- `HUGGINGFACE_TOKEN` or `HF_TOKEN`

## Public API Example

```json
{
  "model": "intfloat/multilingual-e5-small",
  "inputs": ["hello world", "another text"]
}
```

Response includes `model_id`, `dimension`, `vectors`, backend/device metadata, batching state, and timings.

## Architecture

- `backend/genesis_embed_server.py`: FastAPI app, lifespan, frontend serving
- `backend/genesis_embed_server_api.py`: public `/embed`, `/v1/*`, `/health`
- `backend/genesis_embed_server_admin.py`: protected admin endpoints
- `backend/genesis_embed_server_runtime.py`: SentenceTransformers/OpenVINO/PyTorch loading and encoding
- `backend/genesis_embed_server_batching.py`: dynamic batching worker
- `backend/genesis_embed_server_hardware.py`: CUDA, XPU, OpenVINO, NPU/iGPU detection
- `backend/genesis_embed_server_model_manager.py`: Hugging Face cache status/download/delete
- `frontend/src/App.tsx`: admin dashboard and Embedding Lab

## Notes

`torch.compile` is exposed as an optional setting and defaults to `off`. Triton for Windows is not installed by default; OpenVINO is the first-class Intel acceleration path for this app.
