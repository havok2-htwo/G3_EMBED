# G3_EMBED API Documentation

Base URL: `http://127.0.0.1:8777`

## Public Routes

Public embedding routes are open until at least one client API key is configured. Once keys exist, callers must provide:

```http
X-API-Key: g3_embed_...
```

### `GET /health`

Returns service status, default model, loaded models, resolved runtime backend/device, detected hardware, and selected settings.

`media` lets clients discover which media the default model accepts and the current limits (other `/health` fields omitted):

```json
{
  "media": {
    "modalities": ["image", "audio", "video"],
    "limits": {
      "max_images_per_input": 4,
      "max_images_per_request": 16,
      "max_audio_clips_per_input": 2,
      "max_audio_seconds_per_clip": 30,
      "max_audio_seconds_per_request": 120,
      "max_videos_per_request": 1,
      "max_video_frames": 8,
      "max_media_item_mb": 8,
      "max_image_megapixels": 25,
      "max_request_mb": 32,
      "media_pass_budget_mb": 1200,
      "max_concurrent_media_requests": 2,
      "max_video_frame_pixels": 2073600,
      "min_video_frame_side": 8,
      "min_audio_seconds": 0.1,
      "audio_sample_rates": [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000],
      "audio_formats": ["WAV", "FLAC", "OGG", "MP3"],
      "image_formats": ["PNG", "JPEG", "WEBP", "GIF", "BMP"],
      "video_frame_formats": ["PNG", "JPEG", "WEBP"]
    }
  }
}
```

`modalities` lists the default model's media types that are enabled (`image_inputs_enabled` etc.). It is `[]` for text-only models and when the resolved backend is not `pytorch`.

### `POST /embed`

Genesis-compatible embedding route.

Request:

```json
{
  "model": "google/embeddinggemma-2",
  "inputs": ["Text A", "Text B"],
  "input_type": "document"
}
```

`inputs` may also be a single string, a single media object, or a list mixing strings and media objects (see [Image, audio and video inputs](#image-audio-and-video-inputs)). A list holds at most 2048 inputs (HTTP 422 above). `model` defaults to the configured default model (`google/embeddinggemma-2`).

`input_type` is optional: `"query"` or `"document"` prepends the model's query/document prompt (for example `query: ` / `passage: ` for E5, `task: search result | query: ` / `title: none | text: ` for EmbeddingGemma 2). Aliases `passage`, `search_query` and `search_document` are accepted. Omit it, or send any other value, to embed the raw text. For media objects the prompt goes in front of the `text` part only; a media-only object gets no prompt.

Strings are always embedded as text, including strings that look like URLs, file paths or data URIs; the server never fetches or reads them as media.

Response:

```json
{
  "model_id": "google/embeddinggemma-2",
  "dimension": 768,
  "vectors": [[0.1, 0.2]],
  "backend": "pytorch",
  "device": "cuda",
  "device_label": "NVIDIA GeForce RTX 4090",
  "input_count": 1,
  "input_chars": 12,
  "input_tokens": 5,
  "media": null,
  "load_duration_ms": 0,
  "encode_duration_ms": 8.5,
  "total_duration_ms": 9.1,
  "texts_per_second": 117.6,
  "batched": true
}
```

- `input_chars`: characters of all text, prompts included (media not counted).
- `input_tokens`: counted tokens over all inputs (prompt, special and media tokens included). Always set for requests with media. For text-only requests it is set when the lengths were checked (`truncate` false, the default) and `null` when `truncate` is on.
- `media`: `null` for text-only requests, else `{"images": 2, "audio_seconds": 12.5, "video_frames": 8}` (totals over the request; audio after resampling).
- `batched`: always `false` for requests with media (they bypass dynamic batching).

#### Texts over the token limit

By default (`too_long_policy: reject`), a request is rejected when any input, counted with its prompt and special tokens, exceeds the model's token limit (`max_seq_length`, e.g. 4096 for EmbeddingGemma 2, 512 for E5). Nothing is embedded; only that request fails. A model that is not loaded yet is not loaded for a request whose texts its tokenizer already finds too long (catalog models with a known token window).

```json
HTTP 400
{
  "detail": {
    "error": "input_too_long",
    "message": "1 input(s) exceed the token limit of 4096 for 'google/embeddinggemma-2'. Split the text, or send \"truncate\": true to cut it at the limit.",
    "model_id": "google/embeddinggemma-2",
    "max_tokens": 4096,
    "inputs": [{"index": 1, "tokens": 6004}]
  }
}
```

`index` is the position in `inputs`. Send `"truncate": true` to cut texts at the limit instead, or `"truncate": false` to reject even when the server is set to `truncate`.

Inputs with media are never truncated: `truncate` and `too_long_policy` apply only to plain strings (and text-only objects). A media input over the limit always returns `input_too_long`, with the message suggesting fewer images, frames or audio seconds, or shorter text.

#### Image, audio and video inputs

Supported by `google/embeddinggemma-2` on the PyTorch backend. Each media object is one input and returns one vector; vectors come back in input order.

```json
{
  "model": "google/embeddinggemma-2",
  "input_type": "document",
  "inputs": [
    "plain text",
    {"text": "a red bicycle", "image": "iVBORw0KGgoAAAANSUhEUgAA..."},
    {"image": ["data:image/jpeg;base64,/9j/4AAQSkZJRg...", "UklGRlYAAABXRUJQ..."]},
    {"audio": "data:audio/wav;base64,UklGRiQAAABXQVZF..."},
    {"text": "cooking pasta", "video": ["/9j/4AAQ...", "/9j/4AAQ...", "/9j/4AAQ..."]}
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `text` | string, optional | Caption or query text. `input_type` prompts apply to it. |
| `image` | string or list of strings | Base64 or data URI. PNG, JPEG (at most 64 progressive scans), WEBP, GIF (first frame), BMP. EXIF orientation is applied. |
| `audio` | string or list of strings | Base64 or data URI. WAV, FLAC, OGG, MP3 (recognised by the file signature; other containers are rejected before decoding); mono or stereo; 8000, 11025, 16000, 22050, 24000, 32000, 44100 or 48000 Hz; 0.1 s up to `max_audio_seconds_per_clip` (30 s at most). |
| `video` | list of strings | One video as pre-sampled frames (about 1 fps), base64 PNG, JPEG or WEBP, all the same size, at least 8 pixels on each side and at most 1920x1080 pixels. Every frame is embedded. |

- At least one field must be set; `{}` or `{"image": []}` returns 400 `invalid_input`. Unknown keys return 422.
- A list field holds at most 64 entries (HTTP 422 above); the per-input and per-request limits below are lower anyway.
- Base64 is decoded strictly: no whitespace or line breaks. A data URI must use `;base64,`; its media type is not checked.
- Parts are embedded in the order text, images, audio, video, whatever the JSON key order.
- An object with only `text` is handled like a plain string (batched, `truncate` applies).
- Token cost per input: 2 special tokens + text tokens + up to 282 per image (by aspect ratio, not resolution; usually 258-270) + up to 142 per video frame + 25 per audio second plus 2 per clip. The total must fit the token window (`max_seq_length`, default 4096).

curl:

```bash
IMG=$(base64 -w 0 photo.jpg)
curl -s http://127.0.0.1:8777/embed \
  -H "Content-Type: application/json" \
  -d "{\"inputs\": [{\"text\": \"a red bicycle\", \"image\": \"data:image/jpeg;base64,$IMG\"}]}"
```

Response for a media request (excerpt):

```json
{
  "model_id": "google/embeddinggemma-2",
  "dimension": 768,
  "input_count": 4,
  "input_chars": 4,
  "input_tokens": 859,
  "media": {"images": 1, "audio_seconds": 2.0, "video_frames": 4},
  "batched": false
}
```

Requests with media bypass dynamic batching and run in forward passes sized by `media_pass_budget_mb`. They use one of `max_concurrent_media_requests` slots; when all are busy the request returns 429 right away (no queueing).

#### Errors

All error details are objects with `error` and `message` plus the fields below (HTTP 400 unless noted). The request is rejected as a whole; nothing is embedded.

| Status | `error` | Extra fields | When |
|---|---|---|---|
| 400 | `invalid_input` | `index` | Media object without any field |
| 400 | `invalid_media` | `index`, `field`, `position`, `reason` | Bad base64, format not allowed, undecodable data, JPEG with more than 64 scans, audio rate/channels/too short, frame over 1920x1080 pixels or under 8 pixels on a side, frames of different size |
| 400 | `media_disabled` | `modality` | The modality is switched off (`image_inputs_enabled` etc.) |
| 400 | `media_unsupported` | `model_id`, `modality` | Text-only model, or the model does not run on PyTorch |
| 400 | `media_limit_exceeded` | `limit`, `value`, `max`; `index` for per-input limits; `field`, `position` for single items | A count, size, duration or budget limit (see settings) |
| 400 | `reserved_token` | `index`, `token` | A text contains `<\|image\|>`, `<\|audio\|>` or `<\|video\|>` (models with media support only, also plain strings) |
| 400 | `input_too_long` | `model_id`, `max_tokens`, `inputs` | See above |
| 413 | `request_too_large` | `max_bytes` | Body of `POST /embed` or `POST /v1/embeddings` over `max_request_mb`; body of any other POST/PUT/PATCH/DELETE route over 4 MB |
| 429 | `media_busy` | `max_concurrent` | All media slots busy; header `Retry-After: 1` |
| 503 | `gpu_memory_unavailable` | `required_mb`, `available_mb` | Not enough free VRAM for the next media pass (estimate + 256 MB margin); nothing was run |
| 503 | `gpu_out_of_memory` | `model_id` | CUDA ran out of memory during encoding (text or media); the CUDA cache was released |

`limit` in `media_limit_exceeded` is one of `max_images_per_input`, `max_images_per_request`, `max_audio_clips_per_input`, `max_audio_seconds_per_clip`, `max_audio_seconds_per_request`, `max_videos_per_request`, `max_video_frames`, `max_media_item_mb`, `max_image_megapixels` or `media_pass_budget_mb`.

Examples:

```json
HTTP 400
{
  "detail": {
    "error": "invalid_media",
    "message": "Input 1 audio[0]: sample rate 44101 Hz is not allowed (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000).",
    "index": 1,
    "field": "audio",
    "position": 0,
    "reason": "sample rate 44101 Hz is not allowed (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)"
  }
}
```

```json
HTTP 400
{
  "detail": {
    "error": "media_limit_exceeded",
    "message": "Input 0 has 6 images; at most 4 are allowed.",
    "limit": "max_images_per_input",
    "value": 6,
    "max": 4,
    "index": 0
  }
}
```

```json
HTTP 400
{
  "detail": {
    "error": "media_unsupported",
    "message": "Model 'intfloat/multilingual-e5-small' does not accept image inputs here (is text-only).",
    "model_id": "intfloat/multilingual-e5-small",
    "modality": "image"
  }
}
```

```json
HTTP 413
{
  "detail": {
    "error": "request_too_large",
    "message": "Request body exceeds 32 MB (max_request_mb).",
    "max_bytes": 33554432
  }
}
```

```json
HTTP 429  (Retry-After: 1)
{
  "detail": {
    "error": "media_busy",
    "message": "Too many media requests in progress; retry shortly.",
    "max_concurrent": 2
  }
}
```

```json
HTTP 503
{
  "detail": {
    "error": "gpu_memory_unavailable",
    "message": "Not enough free GPU memory for this request (812.4 MB GPU memory available, 1046.0 MB needed); retry later.",
    "required_mb": 1046.0,
    "available_mb": 812.4
  }
}
```

```json
HTTP 503
{
  "detail": {
    "error": "gpu_out_of_memory",
    "message": "GPU out of memory while encoding with 'google/embeddinggemma-2'; retry later or send smaller requests.",
    "model_id": "google/embeddinggemma-2"
  }
}
```

The 413 check uses `Content-Length` before the body is read, and counts bytes for chunked bodies. Before answering, the server reads and drops the rest of an oversized body (up to 4x the limit or 10 s), so the client finishes its upload and receives the 413 instead of a connection reset; a client that sent `Expect: 100-continue` gets the 413 without uploading. Base64 adds about a third to the media size. The same 413 (with a fixed 4 MB limit and no setting name in the message) applies to the admin and login routes.

Request validation errors (HTTP 422, e.g. an unknown key in a media object) list only `loc`, `msg` and `type` per error, at most 50 entries; the submitted input is not echoed back. Because `inputs` is a union, there is one entry per type that did not match. All unknown keys of one media object give a single entry:

```json
HTTP 422
{
  "detail": [
    {"loc": ["body", "inputs", "str"], "msg": "Input should be a valid string", "type": "string_type"},
    {"loc": ["body", "inputs", "MediaInput"], "msg": "Input should be a valid dictionary or object to extract fields from", "type": "model_attributes_type"},
    {"loc": ["body", "inputs", "list[union[str,MediaInput]]", 0, "str"], "msg": "Input should be a valid string", "type": "string_type"},
    {"loc": ["body", "inputs", "list[union[str,MediaInput]]", 0, "MediaInput"], "msg": "Value error, unknown keys 'picture'; allowed: text, image, audio, video", "type": "value_error"}
  ]
}
```

A 500 on a request with media returns only `"detail": "embedding failed"`, without internal error text.

### `GET /v1/models`

OpenAI-compatible model list. Each entry carries `max_model_len` (vLLM's field): the token window `/embed` applies to that model — the native limit of a loaded model, otherwise the catalog limit, capped by the `max_seq_length` setting. Models whose limit is only known after loading leave it out.

### `POST /v1/embeddings`

OpenAI-compatible embedding route.

Request:

```json
{
  "model": "google/embeddinggemma-2",
  "input": ["Text A", "Text B"],
  "input_type": "query"
}
```

`input_type` and `truncate` work as for `POST /embed` (non-standard extensions; OpenAI clients can send them as extra body fields). Texts over the token limit return the same HTTP 400 `input_too_long` error.

`input` accepts the same media objects as `inputs` on `POST /embed`, with the same limits and errors:

```json
{
  "model": "google/embeddinggemma-2",
  "input": [
    "Text A",
    {"text": "a red bicycle", "image": "data:image/png;base64,iVBORw0KGgo..."},
    {"audio": "UklGRiQAAABXQVZF..."}
  ]
}
```

```bash
curl -s http://127.0.0.1:8777/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"google/embeddinggemma-2\", \"input\": [{\"audio\": \"$(base64 -w 0 clip.wav)\"}]}"
```

Response uses OpenAI's `object`, `data`, `model`, and `usage` shape. Embeddings are returned as float arrays. `usage.prompt_tokens` (and `total_tokens`) is the counted `input_tokens` when known (requests with media, and text requests checked against the token limit); otherwise it falls back to the approximation `input_chars // 4`.

API key usage counts inputs (one per string or media object), not tokens.

## Admin Routes

Admin routes use username/password login and an httpOnly session cookie. The default first-run login is `admin` / `admin`, and the password must be changed before protected admin actions are available.

### `POST /api/admin/auth/login`

Creates a browser session.

```json
{
  "username": "admin",
  "password": "admin"
}
```

### `POST /api/admin/auth/change-password`

Changes the current admin password and refreshes the session.

```json
{
  "current_password": "admin",
  "new_password": "new-password"
}
```

### `GET /api/admin/auth/whoami`

Returns the current session user and password-change state.

### `POST /api/admin/auth/logout`

Clears the current browser session.

### `GET /api/admin/api-keys`

Returns metadata for configured client API keys.

### `POST /api/admin/api-keys`

Creates a client API key and returns the plaintext token once.

```json
{
  "alias": "local-client"
}
```

### `DELETE /api/admin/api-keys/{key_id}`

Deletes a client API key.

### `GET /api/admin/settings`

Returns current settings, UI options, model cache state, and runtime status.

### `PUT /api/admin/settings`

Updates settings:

- `default_model`
- `execution_target`
- `gpu_device` (`auto` or `cuda:N`, numbered like nvidia-smi; optional, an omitted value keeps the stored GPU)
- `backend_override`
- `model_cache_path`
- `batching_enabled`
- `batch_wait_time_ms`
- `batch_max_texts`
- `batch_max_chars`
- `encode_batch_size` (texts per forward pass, `0` = model default)
- `max_seq_length` (token limit per text, `0` = model default, clamped to the model's native limit)
- `too_long_policy` (`reject` = HTTP 400 for texts over the limit, default; `truncate` = cut silently)
- `precision`
- `quantization` (`none`, `int8`, `nf4`; PyTorch CUDA only)
- `compile_mode` (`off`, `regional`, `full`)
- `warmup_on_load`
- `huggingface_token`

Optional fields (`encode_batch_size`, `max_seq_length`, `too_long_policy`, `quantization` and all media settings) keep their current value when omitted from the payload, so an older client that does not know a field never resets it.

Media settings (out-of-range values are clamped to `min-max`):

- `image_inputs_enabled`, `audio_inputs_enabled`, `video_inputs_enabled` (default `true`)
- `max_images_per_input` (default 4, 1-8)
- `max_images_per_request` (default 16, 1-32)
- `max_audio_clips_per_input` (default 2, 1-4)
- `max_audio_seconds_per_clip` (default 30, 1-30; longer clips are rejected, never cut)
- `max_audio_seconds_per_request` (default 120, 1-300)
- `max_videos_per_request` (default 1, 1-2; at most one video per input)
- `max_video_frames` (default 8, 1-16; frames per video)
- `max_media_item_mb` (default 8, 1-25; decoded size of one image, audio clip or frame)
- `max_image_megapixels` (default 25, 1-50; read from the image header before decoding)
- `max_request_mb` (default 32, 1-256; body size of `POST /embed` and `POST /v1/embeddings`, HTTP 413 above it)
- `media_pass_budget_mb` (default 1200, 200-4096; estimated extra VRAM per media forward pass; audio clips count as padded to the longest clip of the pass)
- `max_concurrent_media_requests` (default 2, 1-8; HTTP 429 when all slots are busy)

### `GET /api/admin/settings/presets`

Lists the settings presets: the built-in GPU tiers first, in tier order, then saved presets sorted by name. A preset is a set of settings values; applying one is a normal `PUT /api/admin/settings` (the dashboard's "Load into form" fills the form, Save Settings applies it), so a preset never bypasses validation.

Built-in tiers (names identical across the G3 services, read-only):

- `32GB VRAM`: RTX 5090, every G3 service on one card (whole-card peak <= 30 GB)
- `24GB VRAM`: RTX 4090, every G3 service on one card (whole-card peak <= 23 GB)
- `24+8 VRAM`: 3090 (24 GB) + 2070S (8 GB) box, EMBED on the 24 GB card. Its `values.gpu_device` is the CUDA card whose total memory is within 2 GiB of 24 GB, wherever it sits on this machine; with no such card `gpu_device` is left out and `note` says why.

Single-card tiers never set `gpu_device`. There is no `8+24 VRAM` tier for EMBED (it runs on the 3090 next to RERANK).

```json
{
  "presets": [
    {
      "name": "24+8 VRAM",
      "built_in": true,
      "description": "3090 + 2070S box, EMBED on the 3090 (24 GB) next to RERANK. ...",
      "target_vram_gb": 24,
      "note": null,
      "values": { "batch_max_texts": 32, "precision": "auto", "gpu_device": "cuda:0", "...": "..." }
    },
    { "name": "lab", "built_in": false, "description": "", "created_at": "2026-10-16T12:00:00", "note": null, "values": { "batch_max_texts": 64 } }
  ]
}
```

Preset keys (everything else is dropped): `default_model`, `execution_target`, `gpu_device`, `backend_override`, `precision`, `quantization`, `compile_mode`, `cuda_graphs`, `warmup_on_load`, `batching_enabled`, `batch_wait_time_ms`, `batch_max_texts`, `batch_max_chars`, `encode_batch_size`, `encode_token_batching`, `max_seq_length`, `media_pass_budget_mb`, `max_concurrent_media_requests`. The cache path, the Hugging Face token, `too_long_policy` and the per-request media limits are never stored in a preset.

### `PUT /api/admin/settings/presets/{name}`

Saves (or replaces) a preset from `{"values": {...}}`, usually the current settings form. Values are filtered to the preset keys and normalised exactly like saved settings (clamped, invalid options reset). Names are trimmed, inner whitespace collapsed, at most 64 characters. Stored in `logs/genesis_embed_presets.json` next to the settings file. Returns `{"ok": true, "preset": {...}}`.

HTTP 400: empty or too long name, a built-in tier name, or no preset key in `values`.

### `DELETE /api/admin/settings/presets/{name}`

Deletes a saved preset. HTTP 400 for a built-in tier, 404 when no saved preset has that name.

### `GET /api/admin/system`

Returns hardware and runtime status.

### `GET /api/admin/models`

Returns model cache status for the configured or provided storage path.

### `POST /api/admin/models/download`

Starts or resumes a Hugging Face snapshot download.

### `POST /api/admin/models/delete`

Deletes the selected model cache folder.

### `GET /api/admin/stats`

Returns summary metrics, recent embedding history, and recent batches. History entries carry `input_tokens` (when counted) and `media` counts, never media data.

### `GET /api/admin/queue`

Returns the dynamic batching worker snapshot.

### `POST /api/admin/compare`

Embeds two texts and returns vectors plus cosine similarity, dot product, Euclidean distance, timings, and backend/device metadata.

### `POST /api/admin/benchmark`

Runs repeated direct embedding passes for supplied text inputs and returns throughput metrics. At most 2048 inputs (HTTP 422 above).
