# G3_EMBED API Documentation

Base URL: `http://127.0.0.1:8770`

## Public Routes

### `GET /health`

Returns service status, default model, loaded models, resolved runtime backend/device, detected hardware, and selected settings.

### `POST /embed`

Genesis-compatible embedding route.

Request:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "inputs": ["Text A", "Text B"]
}
```

`inputs` may also be a single string.

Response:

```json
{
  "model_id": "intfloat/multilingual-e5-small",
  "dimension": 384,
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
  "model": "intfloat/multilingual-e5-small",
  "input": ["Text A", "Text B"]
}
```

Response uses OpenAI's `object`, `data`, `model`, and `usage` shape. Embeddings are returned as float arrays.

## Admin Routes

All admin routes require:

```http
X-Admin-Key: genesis_embed_admin_...
```

### `GET /api/admin/keys`

Returns metadata for the active persistent admin key.

### `POST /api/admin/keys`

Rotates the persistent admin key and returns the new plaintext token once.

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
- `precision`
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
