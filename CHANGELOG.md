# Changelog

## Unreleased

- Texts over the token limit for a model that is not loaded are rejected by its tokenizer alone (CPU, cached for up to 4 models) instead of loading the model onto the GPU first. Same HTTP 400 `input_too_long` response as before. Clients sending over-limit texts for `intfloat/multilingual-e5-large` had kept it loaded (and compiled) next to the default model, ~2.4 GB of VRAM. Requests that fit still load the model as before. Applies to catalog models with a known native window; the catalog now lists it for E5, BGE M3, MiniLM and Qwen3, so `/v1/models` also reports their `max_model_len` before loading.
- Text forward passes are packed by padded tokens (`encode_token_batching`, default on): budget `encode_batch_size` x token window, at most 128 texts per pass, so short texts share passes while full-window texts keep the old peak VRAM. EmbeddingGemma 2 (`full` compile, RTX 4090): 512 short texts 3.0 s -> 0.16 s, 256 x 1k chars 1.5 s -> 0.26 s.
- CUDA graphs for short EmbeddingGemma 2 text passes (`cuda_graphs`, default on; up to 512 tokens x 128 texts, a few MB of extra VRAM with `full` compile, captured at model load, normal forward for everything else): a single query 16 -> 4.3 ms warm (32-52 ms before this release). Capture adds 1.5-2.5 min to a `full`-compile load (~9 s without compile).
- Media requests on CUDA no longer run `gc.collect()` after a successful pass (only `empty_cache()`); it cost 0.3-1 s per request in a compiled process (1 image 419 -> 80 ms). Failures and OOM still collect.
- sentence-transformers' per-call `model.to(device)` / `model.eval()` return early when nothing changes (~11 ms per call with EmbeddingGemma 2).

- SECURITY: strings sent to `/embed`, `/v1/embeddings` and the admin compare/benchmark routes that looked like media URLs, file paths or data URIs were fetched or read as media by sentence-transformers (SSRF and local file read with EmbeddingGemma 2). Text is now always passed as `{"text": ...}` and embedded as text; embeddings of normal text are unchanged.
- `/embed` and `/v1/embeddings` accept image, audio and video inputs for `google/embeddinggemma-2` (PyTorch backend only): objects `{"text", "image", "audio", "video"}` with base64 or data-URI payloads, one embedding per object, mixable with plain strings. Video is a list of pre-sampled frames (no video file decoding). Prompts apply to the text part only.
- Media limits as admin settings with hard caps: per-modality switches, images per input/request, audio clips and seconds, videos and frames, item size, image megapixels, `media_pass_budget_mb` and `max_concurrent_media_requests`. Listed under `media` in `/health`.
- Media inputs run in forward passes sized by an estimated VRAM budget (1200 MB default; audio clips count as padded to the longest clip of the pass, as the feature extractor does), with a free-VRAM check before each pass on CUDA (HTTP 503 `gpu_memory_unavailable`) and the CUDA cache released after each media request, also after a failed pass once its tensors are freed. CUDA out-of-memory on `/embed` and `/v1/embeddings` (including `CUDA error: out of memory` and cuBLAS/cuDNN allocation failures) now returns HTTP 503 `gpu_out_of_memory` (text and media) instead of 500 and releases the CUDA cache.
- Media inputs are never truncated; over the token window they return HTTP 400 `input_too_long`, also with `"truncate": true`.
- New errors: `invalid_input`, `invalid_media`, `media_disabled`, `media_unsupported`, `media_limit_exceeded`, 429 `media_busy`. 500s on media requests no longer include internal error text.
- With EmbeddingGemma 2, texts containing `<|image|>`, `<|audio|>` or `<|video|>` are rejected with HTTP 400 `reserved_token` (they crashed preprocessing and failed every request in the same batch).
- Request bodies of `/embed` and `/v1/embeddings` are capped by `max_request_mb` (32 MB default), all other POST/PUT/PATCH/DELETE routes (admin, login) at 4 MB: HTTP 413 `request_too_large`. The rest of an oversized body is read and dropped (bounded) first, so clients get the 413 instead of a connection reset.
- HTTP 422 validation errors no longer echo the submitted input and list at most 50 errors. Requests (and admin benchmarks) hold at most 2048 inputs and a media field at most 64 entries (HTTP 422); all unknown keys of a media object give one error. Bodies are validated before authentication, so a small hostile body could otherwise produce millions of errors (GBs of RAM).
- Media decoding hardening: progressive JPEGs with more than 64 scans are rejected before decoding (minutes of CPU otherwise); decoded images above about 2.6 MP and frames above about 1.3 MP are downscaled right away instead of being kept at full resolution; video frames need at least 8 pixels on each side (1 px high frames broke the token count and failed with 500); audio must carry a WAV/FLAC/OGG/MP3 signature before libsndfile parses it.
- `PUT /api/admin/settings`: optional fields left out of the payload keep their current value instead of being reset to the default (an older admin tab can no longer re-enable a disabled modality or reset a lowered limit).
- Responses include `input_tokens` and `media` counts; `/v1/embeddings` `usage` reports counted tokens when known instead of `chars / 4`. History entries record both (never media data).
- `torch.compile` `full` compiles only the language model of multimodal models, so the vision/audio towers stay eager.
- Added `soundfile` and `scipy` dependencies, `requirements-dev.txt` (pytest) and a `tests/` suite (CPU by default, opt-in GPU budget test via `G3_EMBED_GPU_TESTS=1`).

- Texts over the token limit are now rejected with HTTP 400 `input_too_long` (listing input index and token count) instead of being truncated silently. Setting `too_long_policy` (`reject` default, `truncate`) and per-request `truncate` override it. Checked before batching, prompts and special tokens included.

- `GET /v1/models` reports each model's token window as `max_model_len` (as vLLM does), so clients such as Genesis size their inputs to it instead of a 512-token fallback.
- Upgraded to transformers 5.19, sentence-transformers 6.1 and huggingface_hub 1.x; added torchvision and pillow.
- Added `google/embeddinggemma-2` and made it the default model (768 dims, up to 8192 tokens, default window 4096).
- Added optional `input_type` (`query` / `document`, aliases `passage`, `search_query`, `search_document`) to `/embed` and `/v1/embeddings` to apply each model's prompts.
- PyTorch CPU loads models in float32 (transformers 5 would otherwise use the checkpoint's bf16); `precision=bfloat16` is honored.
- `torch.compile` modes `regional` (per-layer, short warmup) and `full` (whole model, highest throughput) replace `reduce-overhead`; both use `dynamic=True`, warm up at model load and fall back to eager on errors. Saved `reduce-overhead` settings map to `off` (on Windows without Triton it never took effect; re-select a mode after upgrading).
- The default model is preloaded (and compiled) in the background at server start if it is already in the local cache.
- `install_torch_backend.py` installs the matching `triton-windows` on Windows CUDA builds (`GENESIS_SKIP_TRITON_WINDOWS=1` skips it). `torch.compile` is skipped when Triton is missing.
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
