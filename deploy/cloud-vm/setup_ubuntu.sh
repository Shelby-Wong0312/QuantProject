#!/usr/bin/env bash
set -euo pipefail

# This script sets up the venv and installs a systemd service
# Run from anywhere; it detects repo root from its own location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

SERVICE_NAME=${SERVICE_NAME:-quantproject}
VENV_DIR=${VENV_DIR:-venv}
PY=${PY:-python3}

echo "[1/4] Creating virtualenv at $VENV_DIR"
$PY -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel

echo "[2/4] Installing requirements (if requirements.txt exists)"
if [[ -f requirements.txt ]]; then
  "$VENV_DIR/bin/pip" install -r requirements.txt
else
  echo "No requirements.txt at repo root. Install your dependencies manually:"
  echo "  $VENV_DIR/bin/pip install <packages>"
fi

echo "[3/4] Installing systemd service $SERVICE_NAME"
TEMPLATE="$ROOT_DIR/deploy/cloud-vm/quantproject.service.template"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

RUN_SH="$ROOT_DIR/deploy/cloud-vm/run.sh"
chmod +x "$RUN_SH" || true
chmod +x "$ROOT_DIR/deploy/cloud-vm/update_and_restart.sh" || true

# Determine sudo usage
if [[ $EUID -ne 0 ]]; then
  SUDO=sudo
else
  SUDO=
fi

TMP_UNIT=$(mktemp)
sed \
  -e "s|{{WORKING_DIR}}|$ROOT_DIR|g" \
  -e "s|{{USER}}|$USER|g" \
  -e "s|{{RUN_SH}}|$RUN_SH|g" \
  "$TEMPLATE" > "$TMP_UNIT"

$SUDO mv "$TMP_UNIT" "$SERVICE_PATH"
$SUDO chmod 644 "$SERVICE_PATH"

echo "[4/4] Reloading and starting service"
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now "$SERVICE_NAME"

echo "Done. Check status and logs with:"
echo "  systemctl status $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"

