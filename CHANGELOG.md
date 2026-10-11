# Changelog

## Unreleased

- Upgraded to transformers 5.19, sentence-transformers 6.1 and huggingface_hub 1.x; added torchvision and pillow.
- Added `google/embeddinggemma-2` and made it the default model (768 dims, up to 8192 tokens, default window 4096).
- Added optional `input_type` (`query` / `document`, aliases `passage`, `search_query`, `search_document`) to `/embed` and `/v1/embeddings` to apply each model's prompts.
- PyTorch CPU loads models in float32 (transformers 5 would otherwise use the checkpoint's bf16); `precision=bfloat16` is honored.
- `torch.compile` is skipped when Triton is not installed.
- Docker first boot no longer pins a default model; the code default applies.
- Quantization now applies only to Qwen3 models.
- Removed `optimum-intel` (no transformers 5.18+ support yet); OpenVINO targets fall back to PyTorch and `/health` reports `openvino_backend_available`.
- `start.bat` installs missing torch packages (e.g. torchvision) for existing venvs via `install_torch_backend.py --only-missing`.
- Fixed `torch.compile` with sentence-transformers 6 (the inner model moved to `.model`).

- Added `Qwen/Qwen3-Embedding-0.6B`, `-4B` and `-8B` to the model catalog.
- Added settings `max_seq_length` (token window), `encode_batch_size` (GPU batch size) and `quantization` (bitsandbytes INT8/NF4).
- Keep GQA models such as Qwen3 on the memory-efficient SDPA kernel on CUDA instead of the math kernel, cutting attention memory per 4096-token sequence from ~3 GB to a few MB.
- Loading a new precision/quantization variant of a model now unloads the previous variant first.
- Added `accelerate` and `bitsandbytes` dependencies.

## 1.0.0

- Initial standalone `G3_EMBED` app.
- Added FastAPI backend, React/Vite admin dashboard, protected admin routes, startup admin key, persistent key rotation, and public embedding APIs.
- Added Genesis-compatible `/embed` and OpenAI-compatible `/v1/embeddings`.
- Added curated embedding model catalog and cache manager.
- Added hybrid hardware routing through PyTorch CUDA/XPU and OpenVINO NPU/GPU/CPU.
- Added dynamic batching, request history, queue metrics, Embedding Lab, and benchmark endpoint.
- Added Windows and Linux/macOS install/start scripts.
