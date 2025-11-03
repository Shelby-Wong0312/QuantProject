#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

SERVICE_NAME=${SERVICE_NAME:-quantproject}

echo "[1/2] Updating repo"
git pull --rebase --autostash || true

echo "[2/2] Restarting service $SERVICE_NAME"
if [[ $EUID -ne 0 ]]; then SUDO=sudo; else SUDO=; fi
$SUDO systemctl restart "$SERVICE_NAME"
echo "Done. Tail logs with: journalctl -u $SERVICE_NAME -f"

