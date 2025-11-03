#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

SERVICE=${SERVICE_NAME:-quantproject}
VENV_DIR=${VENV_DIR:-venv}
PY="$ROOT_DIR/$VENV_DIR/bin/python"

echo '=== QuantProject Quick Health ==='
echo "Time: $(date)"
echo "Host: $(hostname)"
echo

echo '--- .env presence & key hints ---'
if [[ -f .env ]]; then
  echo 'OK: .env found (values not displayed)'
else
  echo 'WARN: .env not found at repo root'
fi
echo

echo '--- Service status ---'
if command -v systemctl >/dev/null 2>&1; then
  systemctl is-enabled "$SERVICE" 2>/dev/null || true
  systemctl is-active "$SERVICE" 2>/dev/null || true
  systemctl show "$SERVICE" -p ActiveState,SubState,ExecMainStatus,Restart,RestartSec 2>/dev/null || true
else
  echo 'systemctl not available'
fi
echo

echo '--- Recent logs (last 80 lines) ---'
if command -v journalctl >/dev/null 2>&1; then
  journalctl -u "$SERVICE" -n 80 --no-pager --output cat || true
else
  echo 'journalctl not available'
fi
echo

echo '--- Python & venv ---'
if [[ -x "$PY" ]]; then
  "$PY" -V || true
else
  echo "WARN: venv python not found at $PY"
  command -v python3 && python3 -V || true
fi
echo

echo '--- Capital.com API connectivity ---'
if [[ -x "$PY" ]]; then
  "$PY" scripts/check_capital_api.py || true
else
  python3 scripts/check_capital_api.py || true
fi
echo

echo '--- Market data (yfinance) ---'
if [[ -x "$PY" ]]; then
  "$PY" scripts/check_market_data.py AAPL || true
else
  python3 scripts/check_market_data.py AAPL || true
fi
echo

echo '--- System resources ---'
uptime || true
free -h || true
df -h / || true
echo

echo '=== End of Quick Health ==='

