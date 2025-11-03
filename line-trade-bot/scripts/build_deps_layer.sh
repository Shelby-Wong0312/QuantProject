#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/terraform/build"
LAYER_DIR="$BUILD_DIR/deps/python"
ZIP_PATH="$BUILD_DIR/deps_layer.zip"

mkdir -p "$LAYER_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install "line-bot-sdk>=2,<3" -t "$LAYER_DIR"

cd "$BUILD_DIR/deps"
rm -f "$ZIP_PATH"
zip -r "$ZIP_PATH" python >/dev/null
echo "Built deps layer: $ZIP_PATH"

