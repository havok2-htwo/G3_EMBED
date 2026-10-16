#!/usr/bin/env bash
set -euo pipefail

# Seed a container-friendly settings file on first boot so models land in the
# mounted /app/models volume instead of the Windows-style ".\models" default.
# The "_first_start_seed" marker tells the server this is still a first start:
# it applies the "24GB VRAM" preset on top of the defaults, keeps the seeded
# cache path and saves the result (backend/genesis_embed_server_presets.py,
# load_settings_or_first_start_preset). Existing settings (e.g. an admin changed
# the cache path) are left untouched.
SETTINGS_FILE="/app/logs/genesis_embed_settings.json"
if [[ ! -f "$SETTINGS_FILE" ]]; then
  mkdir -p /app/logs
  cat > "$SETTINGS_FILE" <<'JSON'
{
  "_first_start_seed": true,
  "model_cache_path": "/app/models"
}
JSON
fi

# Admin access is username/password (default admin/admin, change forced on first
# login). No temporary startup admin key is generated.

exec "$@"
