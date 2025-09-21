# QuantProject

## Overview
QuantProject is an event-driven quantitative trading platform that handles market data ingestion, signal generation, risk management, and live execution. It targets quantitative trading teams and automation engineers who need to orchestrate Capital.com or MT4 connectivity, run systematic strategies, and monitor deployments. Runtime code has been consolidated under `src/quantproject/`, and legacy modules remain as thin shims for compatibility.

## Architecture Map
```
QuantProject/
|-- src/quantproject/    # Application runtime (domain + application layers)
|   |-- core/             # Event types, EventLoop, shared orchestration utilities
|   |-- data_pipeline/    # Market data loaders, caches, MT4 collectors
|   |-- risk/             # Unified risk toolkit (metrics, stop policies, alerts)
|   |-- execution/        # Broker + portfolio wiring for Capital.com
|   |-- strategies/       # Strategy base classes, indicator + ML integrations
|   |-- signals/          # TradingSignal definitions and generators
|   |-- rl_trading/       # PPO agents and environments
|   |-- models/           # Shared ML artefacts (e.g. LSTM predictors)
|-- scripts/
|   `-- diagnostics/      # CLI health checks (Capital API, MT4, etc.)
|-- tests/                # Pytest suites (unit / smoke / integration markers)
|-- docs/, documents/     # Design docs, stage reports, security summaries
|-- infra/                # Terraform & infrastructure automation
|-- line-trade-bot/       # LINE notification sub-project (separate CLI)
|-- examples/             # Scenario demos (post-refactor consolidation)
`-- config/               # Runtime configuration templates, env samples
```

## Quick Start
1. **Clone & install**
   ```powershell
   git clone <repo-url>
   cd QuantProject
   python -m venv .venv
   .venv\Scripts\Activate
   pip install -e .[dev,line,ml]
   ```
2. **Configure secrets**
   ```powershell
   copy .env.example .env
   # Edit .env to include CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_API_PASSWORD, etc.
   ```
3. **Run essential tests**
   ```powershell
   pytest -m "unit" --cov=quantproject --cov-report=term-missing
   pytest tests/smoke/test_smoke_api.py -m smoke -q
   ```
4. **Launch a dry-run orchestration**
   ```powershell
   python live_trading_system.py --dry-run
   ```
   Diagnostics (optional):
   ```powershell
   python scripts/diagnostics/check_capital_api.py
   ```

## Configuration & Secrets
- `.env.example` lists required environment variables; copy to `.env` and supply credentials.
- Additional configuration JSON/Python files live in `config/` (e.g., API templates, live trading settings).
- Never commit real secrets. In production, load from parameter stores (AWS SSM, Vault, etc.).

## Working with Canonical APIs
### Events & Execution
- `from quantproject.core import EventLoop, MarketEvent, SignalEvent`
- `from quantproject.execution import Broker, Portfolio`

### Data Pipeline
- `from quantproject.data_pipeline.data_manager import DataManager`
- MT4 collectors reside in `quantproject.data_pipeline.mt4_data_collector`; use CLI diagnostics under `scripts/diagnostics/` for runtime checks.

### Risk Toolkit
- `from quantproject.risk import EnhancedRiskManager, DynamicStopLoss, RiskMetrics, PositionSizing`
- Shim modules under `quantproject.risk_management` only re-export these names and will eventually be removed.

### Signals & Strategies
- `from quantproject.signals.signal_generator import SignalGenerator`
- Strategies are implemented in `quantproject.strategies/`; ML models in `quantproject.models/` should be reused rather than duplicated.

### PPO / Reinforcement Learning
- `from quantproject.rl_trading.agents import PPOAgent, PPOConfig`
- Legacy training scripts (`TRAIN_PPO_*`, `PPO_*`) should transition to call the canonical agent API.

## Public API Map
| Module               | Key Classes / Functions                                        | Role                                                                                       |
|----------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `quantproject.core`           | `EventLoop`, `MarketEvent`, `SignalEvent`                      | Event bus + event primitives powering backtesting and live orchestration.                 |
| `quantproject.execution`      | `Broker`, `Portfolio`                                          | Capital.com broker integration and portfolio bookkeeping.                                 |
| `quantproject.data_pipeline`  | `DataManager`, `CapitalHistoryLoader`, `DataCache`             | Unified historical/streaming ingestion with caching and MT4 collectors.                   |
| `quantproject.risk`           | `EnhancedRiskManager`, `DynamicStopLoss`, `RiskMetrics`, `PositionSizing` | Comprehensive risk toolkit (scoring, stop policies, alerts).                      |
| `quantproject.signals`        | `SignalGenerator`, `TradingSignal`, `SignalConfig`             | Converts indicator context into structured trading signals for strategies.                |
| `quantproject.strategies`     | `BaseStrategy`, `StrategyManager`, ML strategy adapters        | Abstractions turning data + signals into actionable orders.                               |
| `quantproject.rl_trading`     | `PPOAgent`, `PPOConfig`, `TradingEnvironment`                  | Reinforcement learning agents and environments based on PPO.                              |
| `quantproject.models`         | `LSTMPricePredictor`, other ML artefacts                       | Shared predictive models consumed by strategies.                                          |
| `scripts.diagnostics`| `check_capital_api.py` (MT4 diagnostics forthcoming)           | CLI health checks for external integrations.                                              |

## Testing & Quality
```powershell
pytest -m "unit" --cov=quantproject --cov-report=term-missing
pytest tests/smoke/test_smoke_api.py -m smoke -q
pytest -m integration        # run manually / nightly when needed
black src tests
isort src tests
flake8 src tests
mypy src
```
- Coverage target: >= 80% for `quantproject/`. New code must not reduce coverage.
- Integration tests should mock external services; real network checks live in `scripts/diagnostics/`.
- Known mypy hotspot: `src/connectors/capital_com_api.py:515` contains a malformed logging string-fix before running mypy.

## Coding Guidelines
- Naming: modules/functions/variables use `snake_case`; classes in `PascalCase`; constants in `UPPER_SNAKE_CASE`.
- Logging: leverage `logging` or `loguru`, avoid `print`, and mask credentials (e.g., `api_key[:6] + '...'`).
- Error handling: raise domain-specific exceptions (`BrokerError`, `RiskComputationError`); avoid bare `Exception`.
- Type hints: add typing for new functions; keep `mypy src` clean under Python 3.9 settings.
- Compatibility shims: keep `core/`, `execution/`, `risk_management/`, `data_pipeline/` minimal, emitting `DeprecationWarning` only.

## Before You Code
1. Confirm the change belongs to a canonical module under `quantproject.*`.
2. Install the appropriate extras (`pip install -e .[dev]` plus `[line]`, `[ml]` if required).
3. Avoid legacy imports; update them to the `quantproject.*` namespace.
4. Add configuration keys to `.env.example` / `config/` rather than hardcoding.
5. Add or update unit/smoke/integration tests with correct markers.
6. Run unit + smoke tests (`pytest -m "unit"`, `pytest tests/smoke/...`) and lint/typing tools before committing.
7. Update README/AGENTS/docs to reflect behavioural changes.

## Common Pitfalls & Anti-patterns
- Modifying shim directories (`core/`, `execution/`, `risk_management/`, `data_pipeline/`): they are transitional layers only.
- Adding dependencies to `requirements*.txt`: maintain extras in `pyproject.toml` instead.
- Logging secrets or full tokens: always mask sensitive data.
- Scattering tests across ad-hoc scripts: consolidate under `tests/` with proper markers; keep diagnostics in `scripts/diagnostics/`.
- Directly hitting real APIs in pytest: mock external services; use CLI diagnostics for live connectivity.
- Duplicating strategy/model implementations: reuse components in `quantproject.models/` and canonical modules.

## Version & Migration Notes
- Canonical implementations now live under `quantproject.*`. Legacy modules (`core/*`, `execution/*`, `risk_management/*`, `data_pipeline/*`, etc.) are thin shims and will emit `DeprecationWarning` in an upcoming release. Update imports to `quantproject.*` as soon as possible.
- Operational diagnostics moved from pytest into `scripts/diagnostics/*`. Keep pytest suites focused on `unit`, `smoke`, and `integration` markers.
- PPO-related scripts (`TRAIN_PPO_*`, `PPO_*`, `simplified_ppo_trainer.py`) now wrap `quantproject.rl_trading.cli`. Use `python -m quantproject.rl_trading.cli <profile>` for training workflows, or call `run_training_profile()` from that module.
- Future releases will drop `requirements*.txt` in favour of extras-based installs (`pip install -e .[extras]`). Update CI and deployment pipelines accordingly.

## Contribution
- Fork the repo and work off feature branches (or `refactor/project-cleanup` while the refactor effort is ongoing).
- Install tooling via `pip install -e .[dev]` and enable hooks (`pre-commit install`).
- Ensure `black`, `isort`, `flake8`, `mypy`, unit + smoke tests all pass before submitting a PR.
- Update documentation (README, AGENTS, subsystem docs) to reflect behaviour changes.
- Follow Conventional Commits (`feat(scope): ...`, `fix(scope): ...`).
- Legacy shim directories (`core/*`, `execution/*`, etc.) are read-only; real work happens in `quantproject/*`.
- For major changes, open an issue or discussion thread first.

## FAQ / Troubleshooting
- **`ModuleNotFoundError: core.event`** - Legacy paths were replaced by shims. Update imports to `from quantproject.core import ...`.
- **Capital.com diagnostics fail** - Ensure `.env` contains valid credentials and run `python scripts/diagnostics/check_capital_api.py` for step-by-step output.
- **mypy reports syntax error in the Capital.com connector module** - Fix the malformed logging string (missing `f` prefix/quotes) before rerunning mypy.
- **Tests attempt to hit real APIs** - Integration tests should mock external calls. Use CLI diagnostics for live checks and mark tests with `@pytest.mark.integration`.
- **`pip install -r requirements.txt` fails** - Dependencies now live in `pyproject.toml`. Use `pip install -e .[dev]` and add extras (`[line]`, `[ml]`, etc.) as needed.
- **Shim directories still executing logic** - Shims must only re-export and warn. Move code into the corresponding `quantproject.*` module and emit `DeprecationWarning` from the shim if necessary.
- **Need a quick health check** - Run `pytest tests/smoke/test_smoke_api.py -m smoke -q`. External connectivity checks reside in `scripts/diagnostics/`.

