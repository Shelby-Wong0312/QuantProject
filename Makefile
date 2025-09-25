SHELL := /bin/bash

AWS_REGION ?= ap-northeast-1
TF_DIR := infra/terraform

# SSM parameter names (override via make VAR=...)
LINE_CAT_PARAM ?= /prod/line/CAT
LINE_SECRET_PARAM ?= /prod/line/SECRET
INGEST_TOKEN_PARAM ?= /prod/ingest/TOKEN

.PHONY: deps-layer package tf-init tf-apply deploy clean test

deps-layer:
	@if [ -x line-trade-bot/scripts/build_deps_layer.sh ]; then \
		bash line-trade-bot/scripts/build_deps_layer.sh; \
	else \
		echo "Missing script: line-trade-bot/scripts/build_deps_layer.sh"; \
		echo "On Windows, run: powershell -ExecutionPolicy Bypass -File line-trade-bot/scripts/build_deps_layer.ps1"; \
	fi

package: deps-layer

tf-init:
	cd $(TF_DIR) && terraform init

tf-apply:
	cd $(TF_DIR) && terraform apply -auto-approve \
	  -var "aws_region=$(AWS_REGION)" \
	  -var "line_channel_access_token_param_name=$(LINE_CAT_PARAM)" \
	  -var "line_channel_secret_param_name=$(LINE_SECRET_PARAM)" \
	  -var "ingest_auth_token_param_name=$(INGEST_TOKEN_PARAM)"

deploy: tf-init tf-apply

test:
	pytest -q || true

clean:
	rm -rf $(TF_DIR)/build

report:
	python - <<"PY"
import json
import os
from pathlib import Path

import yaml

from eval.metrics import compute
from eval.reports import write_markdown

config_path = Path(os.environ.get('CONFIG', 'configs/train/ppo_baseline.yaml'))
if config_path.exists():
    cfg = yaml.safe_load(config_path.read_text())
else:
    cfg = {}
symbols = cfg.get('symbols') or cfg.get('env', {}).get('symbols') or []
universe_path = Path(cfg.get('universe', 'configs/universe/capital.yaml'))
bucket_map = {}
if universe_path.exists():
    uni = yaml.safe_load(universe_path.read_text())
    for bucket, info in (uni.get('buckets') or {}).items():
        for sym in info.get('symbols', []):
            bucket_map[sym] = bucket

run_dir = Path('runs/rl3') / os.environ.get('RUN', 'baseline')
returns_file = run_dir / 'returns.json'
weights_file = run_dir / 'weights.json'
equity_file = run_dir / 'equity.json'
returns = json.loads(returns_file.read_text()) if returns_file.exists() else [0.0]
weights = json.loads(weights_file.read_text()) if weights_file.exists() else ([[0.0 for _ in symbols]] if symbols else [[0.0]])
equity = json.loads(equity_file.read_text()) if equity_file.exists() else [0.0]
if not symbols:
    symbols = [f"asset_{i}" for i in range(len(weights[-1]))]
metrics = compute(returns, equity)
write_markdown(run_dir / 'report.md', metrics, symbols)
print(f"Report written to {report_path}")
PY

