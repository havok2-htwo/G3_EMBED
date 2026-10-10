# Changelog

## Unreleased

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
