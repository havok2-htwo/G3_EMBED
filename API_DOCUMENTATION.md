# G3_EMBED API Documentation

Base URL: `http://127.0.0.1:8777`

## Public Routes

Public embedding routes are open until at least one client API key is configured. Once keys exist, callers must provide:

```http
X-API-Key: g3_embed_...
```

### `GET /health`

Returns service status, default model, loaded models, resolved runtime backend/device, detected hardware, and selected settings.

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

`inputs` may also be a single string. `model` defaults to the configured default model (`google/embeddinggemma-2`).

`input_type` is optional: `"query"` or `"document"` prepends the model's query/document prompt (for example `query: ` / `passage: ` for E5, `task: search result | query: ` / `title: none | text: ` for EmbeddingGemma 2). Aliases `passage`, `search_query` and `search_document` are accepted. Omit it, or send any other value, to embed the raw text.

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
  "load_duration_ms": 0,
  "encode_duration_ms": 8.5,
  "total_duration_ms": 9.1,
  "texts_per_second": 117.6,
  "batched": true
}
```

### `GET /v1/models`

OpenAI-compatible model list.

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

`input_type` works as for `POST /embed` (non-standard extension; OpenAI clients can send it as an extra body field).

Response uses OpenAI's `object`, `data`, `model`, and `usage` shape. Embeddings are returned as float arrays.

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
- `backend_override`
- `model_cache_path`
- `batching_enabled`
- `batch_wait_time_ms`
- `batch_max_texts`
- `batch_max_chars`
- `encode_batch_size` (texts per forward pass, `0` = model default)
- `max_seq_length` (token truncation per text, `0` = model default, clamped to the model's native limit)
- `precision`
- `quantization` (`none`, `int8`, `nf4`; PyTorch CUDA only)
- `compile_mode`
- `warmup_on_load`
- `huggingface_token`

### `GET /api/admin/system`

Returns hardware and runtime status.

### `GET /api/admin/models`

Returns model cache status for the configured or provided storage path.

### `POST /api/admin/models/download`

Starts or resumes a Hugging Face snapshot download.

### `POST /api/admin/models/delete`

Deletes the selected model cache folder.

### `GET /api/admin/stats`

Returns summary metrics, recent embedding history, and recent batches.

### `GET /api/admin/queue`

Returns the dynamic batching worker snapshot.

### `POST /api/admin/compare`

Embeds two texts and returns vectors plus cosine similarity, dot product, Euclidean distance, timings, and backend/device metadata.

### `POST /api/admin/benchmark`

Runs repeated direct embedding passes for supplied text inputs and returns throughput metrics.
