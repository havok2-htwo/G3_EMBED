#!/usr/bin/env bash
set -euo pipefail

# Seed a container-friendly settings file on first boot so models land in the
# mounted /app/models volume instead of the Windows-style ".\models" default.
# Existing settings (e.g. an admin changed the cache path) are left untouched.
SETTINGS_FILE="/app/logs/genesis_embed_settings.json"
if [[ ! -f "$SETTINGS_FILE" ]]; then
  mkdir -p /app/logs
  cat > "$SETTINGS_FILE" <<'JSON'
{
  "default_model": "intfloat/multilingual-e5-small",
  "execution_target": "auto",
  "backend_override": "auto",
  "model_cache_path": "/app/models",
  "batching_enabled": true,
  "batch_wait_time_ms": 25,
  "batch_max_texts": 32,
  "batch_max_chars": 64000,
  "precision": "auto",
  "compile_mode": "off",
  "warmup_on_load": false,
  "huggingface_token": ""
}
JSON
fi

# Admin access is username/password (default admin/admin, change forced on first
# login). No temporary startup admin key is generated.

exec "$@"
