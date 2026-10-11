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
  - OpenVINO targets are currently unavailable (see [OpenVINO](#openvino)); they fall back to PyTorch.
- Embedding Lab with two-text vector generation, cosine similarity, dot product, distance, timings, and raw vector copy

## Models

The v1 model catalog is intentionally curated:

- `google/embeddinggemma-2` default (768 dims, 8k context)
- `intfloat/multilingual-e5-small`
- `intfloat/multilingual-e5-base`
- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-multilingual-gemma2` experimental XL option for high-VRAM tests
- `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`, `Qwen/Qwen3-Embedding-8B` long-context (32k) models

Embeddings are normalized. Prompts are only added when a request sets `input_type` (see [Query and Document Prompts](#query-and-document-prompts)).

Switching models changes the vector space (and usually the dimension). Re-embed stored vectors after switching; vectors from different models must not be mixed.

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
  "model": "google/embeddinggemma-2",
  "inputs": ["hello world", "another text"],
  "input_type": "document"
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

- `too_long_policy` ("Texts Over Token Limit"): `reject` (default) answers HTTP 400 `input_too_long` when any text, including its prompt and special tokens, exceeds the token limit; `truncate` cuts it silently. Requests can override this with `"truncate": true/false`. Oversized texts are rejected before batching, so they never fail other clients' requests.
- `max_seq_length` ("Max Tokens / Text"): token limit per text. `0` uses the model default (EmbeddingGemma 2 and Qwen3: 4096, others: their native limit). Values above a model's limit are clamped (E5: 512, EmbeddingGemma 2: 8192).
- `encode_batch_size` ("GPU Batch Size"): texts per forward pass. `0` uses the model default (EmbeddingGemma 2: 4, Qwen3: 2, others: `batch_max_texts`). `batch_max_texts` / `batch_max_chars` only control how queued requests are merged.
- `quantization`: `none`, `int8` or `nf4` via bitsandbytes. Applied only to Qwen3 models on PyTorch CUDA; ignored for other models.

Measured on an RTX 4090 (peak PyTorch allocation; add ~0.5 GB CUDA context). Retrieval quality is nDCG@10 on GermanQuAD (short Wikipedia passages) and a 1000-document GerDaLIR subset (German court decisions, ~3300 tokens), with query/document prompts:

| Model | VRAM 4096 tok, batch 1 | Tokens/s 4096 tok | GermanQuAD | GerDaLIR |
|---|---|---|---|---|
| E5 Large (512 tok max) | 1.1 GB | 157k (512 tok, batch 48) | 94.6 | 34.1 |
| EmbeddingGemma 2 | 1.9 GB | 60k (batch 4) | 95.6 | 62.2 |
| Qwen3 0.6B | 1.7 GB | 44k | 94.2 | 59.9 |
| Qwen3 4B NF4 | 3.4 GB | 11k | 96.2 | 66.7 |
| Qwen3 4B BF16 | 8.4 GB | 12k | 96.5 | 68.0 |
| Qwen3 8B NF4 | 5.5 GB | 7k | 97.3 | 67.8 |
| Qwen3 8B BF16 | 15.1 GB | 7.7k | 97.4 | 67.9 |

At 4096 tokens Qwen3 is saturated at batch 1-2 and EmbeddingGemma 2 at batch 4; larger GPU batches only add VRAM. Embedding cosine vs. BF16: INT8 ~0.998, NF4 ~0.97-0.98.

## Query and Document Prompts

`POST /embed` and `POST /v1/embeddings` accept an optional `input_type`:

- `"query"`: search queries
- `"document"`: stored passages/documents

The server then prepends the model's prompt. Aliases `passage`, `search_query` and `search_document` are accepted. Without `input_type` (or with any other value), texts are embedded unchanged (previous behavior).

| Model | `query` | `document` |
|---|---|---|
| EmbeddingGemma 2 | `task: search result \| query: ` | `title: none \| text: ` |
| E5 small/base/large | `query: ` | `passage: ` |
| Qwen3 | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` | none |
| BGE, MiniLM | none | none |

Use the same `input_type` convention for indexing and searching, and re-embed stored documents if you start using prompts.

## Notes

On CUDA the runtime disables the `enable_gqa` SDPA path in transformers. Otherwise Qwen3 (GQA) falls back to the math kernel and materializes the full attention matrix (~3 GB extra per 4096-token sequence). K/V are expanded instead so the memory-efficient kernel is used; cuDNN attention was avoided because it rebuilds a plan for every new sequence length.

`LostGentoo/Qwen3-Embedding-8B-AWQ` (compressed-tensors) is not included: with transformers 4.x it is decompressed to BF16 at runtime (~18 GB VRAM) and diverged noticeably from the BF16 reference. Use `Qwen/Qwen3-Embedding-8B` with `nf4` or `int8` instead.

## torch.compile

`compile_mode` (PyTorch CUDA/XPU only, default `off`):

- `regional`: compiles each transformer layer once and reuses the graph for all layers. Short warmup, roughly 2.5x faster for EmbeddingGemma 2 on varied-length inputs.
- `full`: compiles the whole model. Highest throughput (roughly 3.5x for EmbeddingGemma 2) but a warmup of about 1-3 minutes on every model load.

Compilation uses `dynamic=True`, so varying text lengths and batch sizes do not trigger recompiles, and it runs as a warmup while the model loads (covering short/long and padded/unpadded batches), not on the first request. A rare remaining input variant may still compile once on first use. Graphs that fail to compile fall back to eager.

At server start the default model is preloaded in the background when it is already downloaded, so a long `full` warmup does not land on the first request. `/health` stays responsive meanwhile; embedding requests wait until the model is ready. Compile is skipped for bitsandbytes-quantized models and when Triton is missing.

Triton: Linux CUDA builds of PyTorch include it. On Windows, `tools/install_torch_backend.py` installs the `triton-windows` release that matches the installed torch minor version (torch 2.11 -> triton 3.7). Set `GENESIS_SKIP_TRITON_WINDOWS=1` to skip it.

## OpenVINO

EmbeddingGemma 2 requires transformers 5.18+. `optimum-intel`, which sentence-transformers needs for `backend="openvino"`, currently supports only transformers < 5.6, so it is not installed. Without it, `backend_override=openvino` and the Intel NPU/iGPU targets fall back to PyTorch (CUDA, XPU or CPU). `/health` reports `openvino_backend_available`.

## Switching Between Versions

Two git tags mark the dependency stacks:

- `openvino-transformers4`: transformers 4.57, OpenVINO backend, E5 and Qwen3 (no EmbeddingGemma 2)
- `gemma2-transformers5`: transformers 5.19, EmbeddingGemma 2 default, no OpenVINO backend

After `git checkout <tag>`, reinstall the matching dependencies (`start.bat` does this automatically when `requirements.txt` changed; otherwise run `install.bat` / `bash ./install.sh`). Saved settings that point to a model the checked-out version does not know fall back to that version's default model.
