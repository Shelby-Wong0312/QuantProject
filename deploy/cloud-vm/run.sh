#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (assumes this script is in deploy/cloud-vm)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present (systemd also loads it, but this supports manual runs)
if [[ -f .env ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

VENV_DIR=${VENV_DIR:-venv}
APP_ENTRY=${APP_ENTRY:-PPO_LIVE_TRADER.py}

PY_BIN="$ROOT_DIR/$VENV_DIR/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  echo "Python not found at $PY_BIN. Did you run setup_ubuntu.sh?" >&2
  exit 1
fi

exec "$PY_BIN" "$APP_ENTRY"

