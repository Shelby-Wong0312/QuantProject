"""
Strategy Interface Types and Data Classes
Defines standard types and interfaces for the trading strategy system
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, List
import pandas as pd


class SignalType(Enum):
    """Standard signal types for trading strategies"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class StrategyStatus(Enum):
    """Strategy execution status"""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


@dataclass
class TradingSignal:
    """
    Standardized trading signal structure

    Attributes:
        symbol: Trading symbol (e.g., 'AAPL', 'EURUSD')
        signal_type: Type of signal (BUY, SELL, HOLD)
        strength: Signal strength/confidence (0.0 to 1.0)
        strategy_name: Name of the strategy generating the signal
        timestamp: When the signal was generated
        price: Price at signal generation
        metadata: Additional signal-specific data
    """

    symbol: str
    signal_type: SignalType
    strength: float
    strategy_name: str
    timestamp: pd.Timestamp
    price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data after initialization"""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be between 0.0 and 1.0, got {self.strength}")

        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")


@dataclass
class StrategyConfig:
    """
    Strategy configuration parameters

    Attributes:
        name: Unique strategy name
        enabled: Whether strategy is active
        weight: Strategy weight in portfolio (0.0 to 1.0)
        risk_limit: Maximum risk exposure percentage
        max_positions: Maximum number of simultaneous positions
        parameters: Strategy-specific parameters
        symbols: List of symbols this strategy trades
    """

    name: str
    enabled: bool = True
    weight: float = 1.0
    risk_limit: float = 0.02  # 2% default risk limit
    max_positions: int = 10
    parameters: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")

        if not 0.0 <= self.risk_limit <= 1.0:
            raise ValueError(f"Risk limit must be between 0.0 and 1.0, got {self.risk_limit}")

        if self.max_positions <= 0:
            raise ValueError(f"Max positions must be positive, got {self.max_positions}")


@dataclass
class StrategyPerformance:
    """
    Strategy performance metrics

    Attributes:
        strategy_name: Name of the strategy
        total_return: Total return percentage
        win_rate: Percentage of winning trades
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum drawdown percentage
        total_trades: Total number of trades executed
        avg_trade_duration: Average trade duration in minutes
        last_updated: Last performance calculation timestamp
    """

    strategy_name: str
    total_return: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    last_updated: Optional[pd.Timestamp] = None


@dataclass
class Position:
    """
    Trading position information

    Attributes:
        symbol: Trading symbol
        size: Position size (positive for long, negative for short)
        entry_price: Entry price
        current_price: Current market price
        timestamp: Position entry timestamp
        strategy_name: Strategy that opened the position
        pnl: Current unrealized P&L
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
    """

    symbol: str
    size: float
    entry_price: float
    current_price: float
    timestamp: pd.Timestamp
    strategy_name: str
    pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_pnl(self, current_price: float):
        """Update position P&L with current price"""
        self.current_price = current_price
        if self.size > 0:  # Long position
            self.pnl = (current_price - self.entry_price) * self.size
        else:  # Short position
            self.pnl = (self.entry_price - current_price) * abs(self.size)


@dataclass
class RiskMetrics:
    """
    Risk management metrics

    Attributes:
        var_95: Value at Risk at 95% confidence level
        expected_shortfall: Expected Shortfall (Conditional VaR)
        beta: Market beta
        volatility: Portfolio volatility
        correlation: Correlation with benchmark
        exposure: Current market exposure
    """

    var_95: float = 0.0
    expected_shortfall: float = 0.0
    beta: float = 1.0
    volatility: float = 0.0
    correlation: float = 0.0
    exposure: float = 0.0
