import pytest


@pytest.mark.smoke
def test_core_execution_shims_available():
    from quantproject.core import MarketEvent, EventType
    from quantproject.execution import Broker, Portfolio

    assert MarketEvent is not None
    assert EventType.MARKET.name == "MARKET"
    assert Broker is not None
    assert Portfolio is not None


@pytest.mark.smoke
def test_risk_and_data_pipeline_imports():
    from quantproject.risk import EnhancedRiskManager, RiskMetrics
    from quantproject.data_pipeline.data_manager import DataManager

    manager = DataManager(use_cache=False)
    assert hasattr(manager, "get_historical_data")
    assert EnhancedRiskManager is not None
    assert RiskMetrics is not None


@pytest.mark.smoke
def test_models_and_signals_imports():
    from quantproject.models.ml_models import LSTMPricePredictor, XGBoostPredictor
    from quantproject.signals.signal_generator import SignalGenerator

    assert LSTMPricePredictor is not None
    assert XGBoostPredictor is not None
    assert SignalGenerator is not None


@pytest.mark.smoke
def test_rl_trading_aliases():
    from quantproject.rl_trading import PPOAgent, PPOTrainer, TradingEnvironment

    assert PPOAgent is not None
    assert PPOTrainer is not None
    assert TradingEnvironment is not None
