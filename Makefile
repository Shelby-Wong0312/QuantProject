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
