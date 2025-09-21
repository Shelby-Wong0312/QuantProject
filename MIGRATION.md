# Migration Guide

## Canonical Imports
Legacy modules (`core.*`, `execution.*`, `risk_management.*`, `data_pipeline.*`, etc.) now act only as shims and raise `DeprecationWarning`. Update to the canonical `quantproject.*` imports:

| Legacy | Canonical |
| --- | --- |
| `from core.event import MarketEvent` | `from quantproject.core import MarketEvent` |
| `from execution.broker import Broker` | `from quantproject.execution import Broker` |
| `from data_pipeline.data_manager import DataManager` | `from quantproject.data_pipeline.data_manager import DataManager` |
| `from data.data_manager import DataManager` | `from quantproject.data_pipeline.data_manager import DataManager` |
| `from risk_management.stop_loss import StopLoss` | `from quantproject.risk.stop_loss import StopLoss` |
| `from signals.signal_generator import SignalGenerator` | `from quantproject.signals import SignalGenerator` |
| `from models.ml_models.lstm_price_predictor import LSTMPricePredictor` | `from quantproject.models.ml_models import LSTMPricePredictor` |
| `from strategies.ml.lstm_predictor import LSTMPredictor` | `from quantproject.models.ml_models import LSTMPredictor` |
| `from rl_trading.ppo_agent import PPOAgent` | `from quantproject.rl_trading import PPOAgent` |
| `from rl_trading.trading_env import TradingEnvironment` | `from quantproject.rl_trading import TradingEnvironment` |
| `from rl_trading.ppo_agent import PPOTrainer` | `from quantproject.rl_trading import PPOTrainer` |

## PPO Utilities
- All `TRAIN_PPO_*`, `PPO_*`, and `simplified_ppo_trainer.py` scripts are thin wrappers over `quantproject.rl_trading.cli`. Run profiles with `python -m quantproject.rl_trading.cli <profile>`.
- Import `run_training_profile` from `quantproject.rl_trading.cli` when scripting customised experiments.

## Diagnostics vs Tests
Operational health checks moved from `tests/check_*.py` to `scripts/diagnostics/*`. Keep pytest suites focused on unit/integration/smoke:
```bash
python scripts/diagnostics/check_capital_api.py
```

## Shim Behaviour
Shims remain for backward compatibility but emit `DeprecationWarning`. Integration tests (`tests/integration/test_backward_compat_shims.py`) ensure they continue to resolve the canonical modules.
