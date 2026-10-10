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
- `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`, `Qwen/Qwen3-Embedding-8B` long-context (32k) models

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

## Token Window, GPU Batch Size and Quantization

- `max_seq_length` ("Max Tokens / Text"): truncation limit per text. `0` uses the model default (Qwen3: 4096, others: their native limit). Values above a model's native limit are clamped, so E5 always stays at 512 tokens.
- `encode_batch_size` ("GPU Batch Size"): texts per forward pass. `0` uses the model default (Qwen3: 2, others: `batch_max_texts`). `batch_max_texts` / `batch_max_chars` only control how queued requests are merged.
- `quantization`: `none`, `int8` or `nf4` via bitsandbytes, PyTorch CUDA only. Ignored on CPU/OpenVINO.

Measured on an RTX 4090 (peak PyTorch allocation; add ~0.5 GB CUDA context):

| Model | Variant | 512 tok, batch 48 | 4096 tok, batch 1 | 4096 tok, batch 4 |
|---|---|---|---|---|
| E5 Large | BF16 | 1.6 GB | n/a (512 max) | n/a |
| Qwen3 0.6B | BF16 | | 1.7 GB | 3.3 GB |
| Qwen3 4B | NF4 | | 3.4 GB | 6.0 GB |
| Qwen3 4B | BF16 | | 8.4 GB | 11.0 GB |
| Qwen3 8B | NF4 | | 5.5 GB | 8.4 GB |
| Qwen3 8B | INT8 | | 8.9 GB | 12.3 GB |
| Qwen3 8B | BF16 | | 15.1 GB | 18.0 GB |

At 4096 tokens the GPU is saturated at batch 1-2; larger GPU batches only add VRAM. Embedding cosine vs. BF16: INT8 ~0.998, NF4 ~0.977.

Qwen3 models use last-token pooling and expect an instruction prefix on queries (`Instruct: <task>\nQuery:<text>`); documents are embedded raw. G3_EMBED does not add prefixes automatically.

## Notes

On CUDA the runtime disables the `enable_gqa` SDPA path in transformers. Otherwise Qwen3 (GQA) falls back to the math kernel and materializes the full attention matrix (~3 GB extra per 4096-token sequence). K/V are expanded instead so the memory-efficient kernel is used; cuDNN attention was avoided because it rebuilds a plan for every new sequence length.

`LostGentoo/Qwen3-Embedding-8B-AWQ` (compressed-tensors) is not included: with transformers 4.x it is decompressed to BF16 at runtime (~18 GB VRAM) and diverged noticeably from the BF16 reference. Use `Qwen/Qwen3-Embedding-8B` with `nf4` or `int8` instead.

`torch.compile` is exposed as an optional setting and defaults to `off`. Triton for Windows is not installed by default; OpenVINO is the first-class Intel acceleration path for this app.
