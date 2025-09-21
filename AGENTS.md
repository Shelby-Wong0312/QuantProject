# Repository Guidelines
## Project Structure & Module Organization
Core trading workflows live in `src/` (data ingestion, orchestration, execution bridges). Surrounding packages add specialized concerns: `data_pipeline/` for sourcing and caching, `execution/` for broker adapters, `monitoring/` for dashboards, `strategies/` for signal definitions, and `core/` for shared utilities. Infrastructure-as-code rests in `infra/terraform`; operational helpers stay in `scripts/`. Generated outputs and notebooks belong in `reports/`, `data/`, or `documents/`, while transient artifacts should remain in `cache/` or `temp/`. Tests live under `tests/` mirroring module paths, with shared fixtures in `tests/fixtures/`.

## Build, Test, and Development Commands
- `python -m venv .venv && .venv\Scripts\activate`: bootstrap the project environment.
- `pip install -r requirements-dev.txt`: pull runtime, linting, and test dependencies.
- `pytest -q`: run the full suite; `pytest -m "not slow"` focuses on unit checks.
- `pytest --cov=src --cov-report=term-missing`: capture coverage prior to review.
- `make tf-init` / `make tf-apply`: manage AWS resources defined in `infra/terraform`.
- `python -m src.main` or `quant-trade`: quick smoke of the orchestration path.

## Coding Style & Naming Conventions
- Format with `black` (100 cols) then `isort` on `src` and `tests`.
- Lint with `flake8`; type-check critical paths via `mypy src`.
- Stick to 4-space indents, snake_case functions, PascalCase classes, and UPPER_SNAKE_CASE constants.
- Prefer structured `loguru` logging and centralize configuration in `config/` or `.env`.

## Testing Guidelines
- Place tests in `tests/<domain>/test_<unit>.py`; reuse fixtures under `tests/fixtures/`.
- Mark scope with `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow` to align with CI filters.
- Cover new strategies or scripts and attach updated artifacts in `reports/` when visuals shift.
- Stub or record external calls to keep runs deterministic across environments.

## Commit & Pull Request Guidelines
- Follow conventional commits (`feat(trader): ...`, `fix(monitor): ...`) with scopes matching package names.
- Keep commits focused, include test updates, and avoid mixing refactors with features.
- PRs must outline the change, link issues (`Fixes #123`), capture risk or rollback notes, and list test commands.
- Add screenshots or logs for monitoring and trading UX adjustments.

## Security & Configuration Tips
- Copy `.env.example` to `.env`; never commit secrets or broker tokens.
- Run `terraform plan` inside `infra/terraform` and attach the plan when altering infrastructure.
- Rotate shared credentials immediately and prefer parameter stores over flat files.
- Ignore new generated artifacts by extending `.gitignore` early.
