#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

fail() {
  echo "FEHLER: $1" >&2
  exit 1
}

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  bash "$PROJECT_ROOT/install.sh"
fi

[[ -x "$VENV_PYTHON" ]] || fail "Die Installation ist noch nicht abgeschlossen. Bitte zuerst 'bash ./install.sh' ausfuehren."

# Admin access is username/password (default admin/admin, change forced on first login).
echo "Nutze lokale venv unter \"$VENV_DIR\" ..."
echo "Starte G3_EMBED auf http://127.0.0.1:8777 ..."
"$VENV_PYTHON" -m backend.genesis_embed_server
