#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

fail() {
  echo "FEHLER: $1" >&2
  exit 1
}

choose_python() {
  local candidate
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

command -v npm >/dev/null 2>&1 || fail "npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren."
PYTHON_BOOTSTRAP="$(choose_python)" || fail "Kein Python-Interpreter gefunden."

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Erstelle virtuelle Umgebung in \"$VENV_DIR\" ..."
  "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR" || fail "Konnte keine lokale venv erstellen."
fi

echo "Aktualisiere pip, setuptools und wheel ..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel || fail "pip-Update fehlgeschlagen."

echo "Installiere passende PyTorch-Backend-Wheels ..."
"$VENV_PYTHON" "$PROJECT_ROOT/tools/install_torch_backend.py" --python "$VENV_PYTHON" || fail "PyTorch-Backend-Installation fehlgeschlagen."

echo "Installiere Python-Abhaengigkeiten ..."
"$VENV_PIP" install -r requirements.txt || fail "Installation aus requirements.txt fehlgeschlagen."

echo "Installiere Frontend-Abhaengigkeiten ..."
(
  cd "$FRONTEND_DIR"
  npm install
) || fail "npm install fehlgeschlagen."

echo "Baue Frontend ..."
(
  cd "$FRONTEND_DIR"
  npm run build
) || fail "Frontend-Build fehlgeschlagen."

echo "Installation abgeschlossen. Starten mit: bash ./start.sh"
