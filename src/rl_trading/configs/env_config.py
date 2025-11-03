"""
Environment configuration for RL trading
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class TradingEnvConfig:
    """Complete configuration for trading environment"""

    # Market data
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    data_path: Optional[Path] = None

    # Environment parameters
    initial_capital: float = 10000
    max_steps_per_episode: int = 252  # One trading year
    window_size: int = 50  # Historical window for observations

    # State space configuration
    state_features: Dict[str, bool] = field(
        default_factory=lambda: {
            "price_features": True,
            "volume_features": True,
            "technical_indicators": True,
            "lstm_predictions": True,
            "sentiment_analysis": True,
            "portfolio_state": True,
            "time_features": True,
        }
    )

    # LSTM integration
    lstm_horizons: List[int] = field(default_factory=lambda: [1, 5, 20])
    lstm_model_path: Optional[Path] = None

    # Sentiment integration
    sentiment_window_hours: int = 24
    finbert_model_path: Optional[Path] = None

    # Action space
    action_type: str = "discrete"  # 'discrete' or 'continuous'
    discrete_action_levels: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0]
    )
    allow_short_selling: bool = False

    # Reward configuration
    reward_type: str = "risk_adjusted"  # 'simple_pnl' or 'risk_adjusted'
    risk_penalties: Dict[str, float] = field(
        default_factory=lambda: {"volatility": 0.1, "drawdown": 0.2, "exposure": 0.05}
    )

    # Transaction costs
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%

    # Risk management
    max_position_size: float = 0.3  # 30% of portfolio
    max_portfolio_exposure: float = 1.0  # 100% invested
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_drawdown: float = 0.10  # 10% drawdown limit

    # Training configuration
    random_start: bool = True
    train_test_split: float = 0.8
    validation_split: float = 0.1

    # Vectorized environment
    n_envs: int = 4  # Number of parallel environments

    # Logging and debugging
    log_level: str = "INFO"
    save_trades: bool = True
    render_mode: str = "none"  # 'human', 'system', or 'none'


# Preset configurations
PRESETS = {
    "conservative": TradingEnvConfig(
        max_position_size=0.2,
        max_portfolio_exposure=0.8,
        risk_penalties={"volatility": 0.2, "drawdown": 0.3, "exposure": 0.1},
        discrete_action_levels=[0.1, 0.25, 0.5, 0.75],
    ),
    "aggressive": TradingEnvConfig(
        max_position_size=0.5,
        max_portfolio_exposure=1.0,
        allow_short_selling=True,
        risk_penalties={"volatility": 0.05, "drawdown": 0.1, "exposure": 0.02},
        discrete_action_levels=[0.25, 0.5, 0.75, 1.0],
    ),
    "day_trading": TradingEnvConfig(
        max_steps_per_episode=390,  # One trading day (6.5 hours * 60 minutes)
        window_size=30,
        commission_rate=0.0005,  # Lower commission for frequent trading
        max_position_size=0.25,
        discrete_action_levels=[0.25, 0.5, 1.0],  # Fewer levels for faster decisions
    ),
    "swing_trading": TradingEnvConfig(
        max_steps_per_episode=60,  # ~3 months
        window_size=100,
        lstm_horizons=[5, 20, 60],
        max_position_size=0.4,
        discrete_action_levels=[0.33, 0.66, 1.0],
    ),
}


def get_config(preset: Optional[str] = None, **kwargs) -> TradingEnvConfig:
    """
    Get environment configuration

    Args:
        preset: Preset name ('conservative', 'aggressive', 'day_trading', 'swing_trading')
        **kwargs: Override parameters

    Returns:
        TradingEnvConfig instance
    """
    if preset and preset in PRESETS:
        config = PRESETS[preset]
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = TradingEnvConfig(**kwargs)

    return config
