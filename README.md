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
- Image, audio and video (frame list) inputs for EmbeddingGemma 2, with configurable limits and a per-pass VRAM budget (see [Image, Audio and Video Inputs](#image-audio-and-video-inputs))

## Models

The v1 model catalog is intentionally curated:

- `google/embeddinggemma-2` default (768 dims, 8k context), multimodal: text, image, audio and video in one vector space
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

Response includes `model_id`, `dimension`, `vectors`, backend/device metadata, batching state, timings, `input_tokens` and (for media requests) `media` counts.

Image, audio and video inputs are objects in the same list (see [Image, Audio and Video Inputs](#image-audio-and-video-inputs) and `API_DOCUMENTATION.md`).

## Architecture

- `backend/genesis_embed_server.py`: FastAPI app, lifespan, frontend serving
- `backend/genesis_embed_server_api.py`: public `/embed`, `/v1/*`, `/health`, media validation pipeline, request body limit
- `backend/genesis_embed_server_admin.py`: protected admin endpoints
- `backend/genesis_embed_server_runtime.py`: SentenceTransformers/OpenVINO/PyTorch loading and encoding, media chunking
- `backend/genesis_embed_server_media.py`: image/audio/frame decoding, media limits, token formula, VRAM cost estimate and chunk planner (no model, no GPU)
- `backend/genesis_embed_server_batching.py`: dynamic batching worker
- `backend/genesis_embed_server_hardware.py`: CUDA, XPU, OpenVINO, NPU/iGPU detection
- `backend/genesis_embed_server_model_manager.py`: Hugging Face cache status/download/delete
- `frontend/src/App.tsx`: admin dashboard and Embedding Lab
- `tests/`: pytest suite (CPU only by default; see [Tests](#tests))

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

## Image, Audio and Video Inputs

`google/embeddinggemma-2` embeds images, audio and video into the same vector space as text. Media are accepted only for models whose catalog entry declares them (today only EmbeddingGemma 2) and only when the request runs on the PyTorch backend; otherwise the request fails with HTTP 400 `media_unsupported`. All other models stay text-only.

### Input format

`inputs` (`/embed`) and `input` (`/v1/embeddings`) accept strings, objects, or a list mixing both. Each object is one input and produces one embedding:

```json
{
  "model": "google/embeddinggemma-2",
  "input_type": "query",
  "inputs": [
    "plain text, as before",
    {"text": "a red bicycle", "image": "iVBORw0KGgo..."},
    {"image": ["data:image/jpeg;base64,/9j/4AAQ...", "UklGR..."]},
    {"audio": "UklGRiQ..."},
    {"text": "what happens here", "video": ["/9j/4AAQ...", "/9j/4AAQ...", "/9j/4AAQ..."]}
  ]
}
```

- `text`: optional string. `image` and `audio`: one base64 string or a list. `video`: one video per object, given as a list of base64 frames. Unknown keys are rejected (HTTP 422); an object needs at least one field (else HTTP 400 `invalid_input`).
- Base64 is decoded strictly: plain base64 or a data URI `data:<type>;base64,<payload>`. Line-wrapped base64 (newlines in the payload) is rejected, so encode with `base64 -w 0` or equivalent.
- The parts of one object are embedded in the order text, images, audio, video, regardless of the key order in the JSON.
- `input_type` prompts are prepended to the `text` part only. A media-only object gets no prompt.
- An object with only `text` behaves like a plain string (batched, `truncate` policy applies).
- Images: PNG, JPEG, WEBP, GIF (first frame) and BMP. EXIF orientation is applied, then the image is converted to RGB. The processor resizes every image, so resolution changes neither tokens nor VRAM much. Images above about 2.6 megapixels (4x the processor's target size) are downscaled right after decoding, so a request never keeps full-resolution pixels in RAM. Progressive JPEGs with more than 64 scans are rejected (`invalid_media`) before decoding: decode time grows with the scan count.
- Audio: WAV, FLAC, OGG and MP3, recognised by their file signature (other containers such as AIFF or CAF are rejected before the audio library parses them), mono or stereo (stereo is downmixed), sample rates 8000, 11025, 16000, 22050, 24000, 32000, 44100 or 48000 Hz (resampled to 16 kHz), 0.1 s to `max_audio_seconds_per_clip` (at most 30 s). Longer clips are rejected, never cut.
- Video: there is no video file decoding. Clients sample the video themselves (about 1 frame per second works well) and send the frames as PNG, JPEG or WEBP. All frames of one video must have the same size, at least 8 pixels on each side and at most 1920x1080 pixels (2,073,600 pixels, either orientation). Frames above about 1.3 megapixels are downscaled after decoding. Every frame sent is embedded; the server does not re-sample.
- Strings are always embedded as text. A string that looks like a URL, file path or data URI is never fetched or read as media.
- With EmbeddingGemma 2, any text (plain string or `text` field) containing `<|image|>`, `<|audio|>` or `<|video|>` is rejected with HTTP 400 `reserved_token`.

### Token cost

Every input counts 2 special tokens plus its text tokens (prompt included) plus:

| Part | Tokens |
|---|---|
| Image | up to 280 plus 2, depending on aspect ratio, not resolution (1024x1024: 258, 640x480: 268, 1920x1080: 266; fewer for extreme aspect ratios) |
| Video frame | up to 140 plus 2 per frame (640x480: 132, 1280x720: 122) |
| Audio | 25 per second plus 2 per clip (30 s: 752) |

The total per input must fit the token window (`max_seq_length`, default 4096 for EmbeddingGemma 2). Inputs with media are never truncated, even with `"truncate": true` or `too_long_policy: truncate`, because cut media tokens break the model's preprocessing; they are rejected with HTTP 400 `input_too_long` instead. Plain strings in the same request still follow the truncate setting. Responses report the counted tokens in `input_tokens` (and in `usage.prompt_tokens` on `/v1/embeddings`).

### Limits

Admin settings (dashboard block "Media limits (image / audio / video)", or `PUT /api/admin/settings`, where a field left out keeps its current value); out-of-range values are clamped to the range below. `/health` publishes the current values under `media.limits`.

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `image_inputs_enabled` / `audio_inputs_enabled` / `video_inputs_enabled` | `true` | | Allow the modality (else HTTP 400 `media_disabled`) |
| `max_images_per_input` | 4 | 1-8 | Images in one input object |
| `max_images_per_request` | 16 | 1-32 | Images over the whole request |
| `max_audio_clips_per_input` | 2 | 1-4 | Audio clips in one input object |
| `max_audio_seconds_per_clip` | 30 | 1-30 | Seconds per clip (the model cuts at 30 s; the server rejects instead) |
| `max_audio_seconds_per_request` | 120 | 1-300 | Audio seconds over the whole request |
| `max_videos_per_request` | 1 | 1-2 | Videos over the whole request (always at most 1 per input) |
| `max_video_frames` | 8 | 1-16 | Frames per video (16 x 142 + 2 = 2274 tokens fits the 4096 window) |
| `max_media_item_mb` | 8 | 1-25 | Decoded size of one image, audio clip or frame; checked on the base64 length before decoding |
| `max_image_megapixels` | 25 | 1-50 | Image size read from the header before the pixels are loaded (decompression bomb guard) |
| `max_request_mb` | 32 | 1-256 | HTTP body size of `POST /embed` and `POST /v1/embeddings` (HTTP 413) |
| `media_pass_budget_mb` | 1200 | 200-4096 | Estimated extra VRAM per media forward pass |
| `max_concurrent_media_requests` | 2 | 1-8 | Media requests decoding/encoding at once (HTTP 429 when full) |

Counts and base64 sizes are checked before anything is decoded, image pixel counts and audio durations from the file headers before the pixels or samples are read, and token counts after decoding. A numeric limit only fails the request that exceeds it: HTTP 400 `media_limit_exceeded` with the limit name, the value and the maximum (`max_request_mb` answers HTTP 413, `max_concurrent_media_requests` HTTP 429).

Fixed caps: a request holds at most 2048 inputs and a media field at most 64 entries (HTTP 422 above, checked before any per-item validation, so a hostile body cannot fan out into millions of validation errors; a 422 lists at most 50 errors). Every other POST/PUT/PATCH/DELETE route (admin, login) takes a body of at most 4 MB (HTTP 413). An oversized body is read and dropped (up to 4x the limit or 10 s) before the 413 is sent, so clients see the 413 instead of a connection reset.

### VRAM

The RTX 4090 is shared with G3_DIA, G3_WHISPER and G3_OMNIVOICE, and often only 3-5 GB are free. The limits keep a media request inside a fixed per-pass budget instead of letting large requests grab whatever is free.

Extra VRAM per forward pass, measured above the loaded model (bf16, SDPA, eager without `torch.compile`, RTX 4090):

| Pass | Tokens per input | Peak extra VRAM |
|---|---|---|
| 1 image (1024x768 or 4000x3000) | 270 | 82 MB |
| 8 images in one input | 2146 | 599 MB |
| 16 single-image inputs | 270 | 1185 MB |
| 30 s audio | 754 | 105 MB |
| 5 x 30 s audio in one input | 3762 | 481 MB |
| Video, 16 frames 1280x720 | 1954 | 574 MB |
| Text, 3975 tokens, batch 1 | 3975 | 480 MB |
| Text, 3975 tokens, batch 4 | 3975 | 1892 MB |

From these the server estimates each pass as `max(74 MB x images + 36 MB x frames + 3.5 MB x audio clips x longest clip seconds, 0.125 MB x rows x longest input tokens) + 50 MB` (the towers and the language model peak at different times). The audio feature extractor pads every clip of a pass to the longest one, so one 30 s clip next to many 0.1 s clips costs like that many 30 s clips; the planner splits such passes. The constants live in the model's `media_cost_mb` entry in `backend/genesis_embed_server_globals.py`.

How a media request runs:

- Requests with media bypass dynamic batching. Media inputs are split, in input order, into forward passes whose estimate stays within `media_pass_budget_mb`; plain strings in the same request use the normal text path. Vectors come back in input order.
- A single input whose own estimate exceeds the budget is rejected with HTTP 400 `media_limit_exceeded` (`limit: media_pass_budget_mb`). With the defaults the largest possible input is estimated at about 845 MB.
- Before each media pass on CUDA, free memory (plus cached but unused memory) must cover the estimate plus a 256 MB margin; otherwise the pass is not attempted and the request returns HTTP 503 `gpu_memory_unavailable`.
- A CUDA out-of-memory error (media or text; also `CUDA error: out of memory` and cuBLAS/cuDNN allocation failures) releases the CUDA cache and returns HTTP 503 `gpu_out_of_memory`.
- After every request that ran media on CUDA, the CUDA cache is released so reserved memory goes back to the other services. When a pass fails, the cache is released only after the failed pass's tensors are freed, so its memory is returned too.
- All passes share the per-model encode lock, so only one forward pass runs at a time.

The budget covers media passes only. Text keeps `encode_batch_size` (EmbeddingGemma 2 default 4: about 1.9 GB above the loaded model at 4 x 4096 tokens, see the last row).

Decoded images and frames stay in CPU RAM until the request finishes, downscaled to at most about 2.6 (images) or 1.3 (frames) megapixels: under 8 MB of RGB each, per concurrent media request.

## Notes

On CUDA the runtime disables the `enable_gqa` SDPA path in transformers. Otherwise Qwen3 (GQA) falls back to the math kernel and materializes the full attention matrix (~3 GB extra per 4096-token sequence). K/V are expanded instead so the memory-efficient kernel is used; cuDNN attention was avoided because it rebuilds a plan for every new sequence length.

`LostGentoo/Qwen3-Embedding-8B-AWQ` (compressed-tensors) is not included: with transformers 4.x it is decompressed to BF16 at runtime (~18 GB VRAM) and diverged noticeably from the BF16 reference. Use `Qwen/Qwen3-Embedding-8B` with `nf4` or `int8` instead.

## torch.compile

`compile_mode` (PyTorch CUDA/XPU only, default `off`):

- `regional`: compiles each transformer layer once and reuses the graph for all layers. Short warmup, roughly 2.5x faster for EmbeddingGemma 2 on varied-length inputs.
- `full`: compiles the whole model. Highest throughput (roughly 3.5x for EmbeddingGemma 2) but a warmup of about 1-3 minutes on every model load.

For multimodal models (EmbeddingGemma 2), `full` compiles only the language model; the vision and audio towers stay eager, so media requests do not trigger recompiles or compile-time VRAM spikes. `regional` compiles only the language model's layers as well. The media VRAM figures above were measured without compile; media VRAM under `full` has not been measured.

Compilation uses `dynamic=True`, so varying text lengths and batch sizes do not trigger recompiles, and it runs as a warmup while the model loads (covering short/long and padded/unpadded batches), not on the first request. A rare remaining input variant may still compile once on first use. Graphs that fail to compile fall back to eager.

At server start the default model is preloaded in the background when it is already downloaded, so a long `full` warmup does not land on the first request. `/health` stays responsive meanwhile; embedding requests wait until the model is ready. Compile is skipped for bitsandbytes-quantized models and when Triton is missing.

Triton: Linux CUDA builds of PyTorch include it. On Windows, `tools/install_torch_backend.py` installs the `triton-windows` release that matches the installed torch minor version (torch 2.11 -> triton 3.7). Set `GENESIS_SKIP_TRITON_WINDOWS=1` to skip it.

## OpenVINO

EmbeddingGemma 2 requires transformers 5.18+. `optimum-intel`, which sentence-transformers needs for `backend="openvino"`, currently supports only transformers < 5.6, so it is not installed. Without it, `backend_override=openvino` and the Intel NPU/iGPU targets fall back to PyTorch (CUDA, XPU or CPU). `/health` reports `openvino_backend_available`. Image, audio and video inputs always need the PyTorch backend.

## Switching Between Versions

Two git tags mark the dependency stacks:

- `openvino-transformers4`: transformers 4.57, OpenVINO backend, E5 and Qwen3 (no EmbeddingGemma 2)
- `gemma2-transformers5`: transformers 5.19, EmbeddingGemma 2 default, no OpenVINO backend

After `git checkout <tag>`, reinstall the matching dependencies (`start.bat` does this automatically when `requirements.txt` changed; otherwise run `install.bat` / `bash ./install.sh`). Saved settings that point to a model the checked-out version does not know fall back to that version's default model.

## Tests

```bat
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe -m pytest tests -q
```

The suite runs on CPU and never loads a model onto the GPU. API tests use a fake model; they do not write to `logs/`. `tests/test_processor_tokens.py` compares the token formula with the real EmbeddingGemma 2 processor on CPU and is skipped when the model is not in `models/`.

`tests/test_gpu_budget.py` is opt-in. It loads EmbeddingGemma 2 (bf16, eager) on CUDA with the process capped at 4 GiB, sends allowed media requests whose planned passes come close to the budget (at least 85%: 15-image passes, 300 s of audio as 4+4+2 clips, one 30 s clip padding many short ones, 16 frames plus 7 images, two text+image rows near 4096 tokens, all modalities in one input) through the real API and encode path, and asserts that the peak stays within `media_pass_budget_mb` and that reserved memory is released afterwards. On CPU it only checks (with the tokenizer) that each case still plans a near-budget pass. It needs at least 3 GiB of free VRAM and shares the GPU with any running services, so run it only when that is acceptable:

```bat
set "G3_EMBED_GPU_TESTS=1" && venv\Scripts\python.exe -m pytest tests/test_gpu_budget.py -q -s
```

PowerShell: `$env:G3_EMBED_GPU_TESTS = "1"; venv\Scripts\python.exe -m pytest tests/test_gpu_budget.py -q -s`. Linux/macOS: `G3_EMBED_GPU_TESTS=1 venv/bin/python -m pytest tests/test_gpu_budget.py -q -s`.
