# Refactor Log – Canonical Package Migration
Date: 2024-09-19 (branch `refactor/project-cleanup`)

1. README rewritten into an LLM-friendly manual (overview, quick start, API map, testing, FAQ).
2. Added `quantproject/` package under `src/`; legacy `core/*`, `execution/*`, `risk_management/*`, `data_pipeline/*` converted to shims with warnings.
3. Moved Capital diagnostics out of pytest (`tests/check_*.py` -> `scripts/diagnostics/`).
4. Updated `pyproject.toml` packages/extras documentation for the new layout.
5. Next steps: consolidate remaining duplicated modules into `quantproject.*`, adjust CI to install extras, and remove shims once consumers migrate.
