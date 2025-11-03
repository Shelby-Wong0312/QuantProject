"""
Base Strategy Class - Abstract base class for all trading strategies
Provides standardized interface and common functionality for trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from .strategy_interface import (
    TradingSignal,
    SignalType,
    StrategyConfig,
    StrategyPerformance,
    Position,
    RiskMetrics,
    StrategyStatus,
)
from ..risk.risk_manager_enhanced import EnhancedRiskManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies

    Provides standardized interface for:
    - Signal generation
    - Position sizing
    - Risk management
    - Performance tracking
    """

    def __init__(self, config: StrategyConfig, initial_capital: float = 100000):
        """
        Initialize base strategy

        Args:
            config: Strategy configuration object
            initial_capital: Initial capital for risk management
        """
        self.config = config
        self.name = config.name
        self.status = StrategyStatus.INACTIVE
        self.initial_capital = initial_capital

        # Initialize strategy state
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.performance = StrategyPerformance(strategy_name=self.name)
        self.risk_metrics = RiskMetrics()

        # Initialize integrated risk manager
        self.risk_manager = EnhancedRiskManager(
            initial_capital=initial_capital,
            max_daily_loss=config.risk_limit,
            max_position_loss=config.risk_limit / 2,
            max_drawdown=config.risk_limit * 5,
            max_leverage=2.0,
        )

        # Performance tracking
        self.trades_history: List[Dict] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = [initial_capital]

        # Initialize parameters
        self._initialize_parameters()

        # Cache for performance optimization
        self._cache = {}

        logger.info(
            f"Initialized strategy: {self.name} with capital: ${initial_capital:,.0f}"
        )

    @abstractmethod
    def _initialize_parameters(self) -> None:
        """
        Initialize strategy-specific parameters
        Override this method in concrete strategy implementations
        """
        pass

    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Calculate trading signals based on market data

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['open', 'high', 'low', 'close', 'volume', 'timestamp']

        Returns:
            List of TradingSignal objects
        """
        pass

    @abstractmethod
    def get_position_size(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> float:
        """
        Calculate position size for a given signal

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price

        Returns:
            Position size (positive for long, negative for short)
        """
        pass

    @abstractmethod
    def apply_risk_management(
        self, position: Position, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Apply risk management rules to a position

        Args:
            position: Current position
            market_data: Current market data

        Returns:
            Dictionary with risk management actions:
            {
                'action': 'hold|close|reduce',
                'new_size': float,
                'reason': str,
                'stop_loss': float,
                'take_profit': float
            }
        """
        pass

    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate trading signal

        Args:
            signal: Trading signal to validate

        Returns:
            True if signal is valid
        """
        try:
            # Basic validation
            if not isinstance(signal, TradingSignal):
                logger.error(f"{self.name}: Invalid signal type")
                return False

            if signal.strength < 0 or signal.strength > 1:
                logger.error(f"{self.name}: Invalid signal strength: {signal.strength}")
                return False

            if signal.signal_type not in SignalType:
                logger.error(f"{self.name}: Invalid signal type: {signal.signal_type}")
                return False

            # Strategy-specific validation
            return self._validate_signal_custom(signal)

        except Exception as e:
            logger.error(f"{self.name}: Error validating signal: {e}")
            return False

    def _validate_signal_custom(self, signal: TradingSignal) -> bool:
        """
        Custom signal validation (override in concrete strategies)

        Args:
            signal: Trading signal to validate

        Returns:
            True if signal passes custom validation
        """
        return True

    def update_performance(self, trades: List[Dict]) -> None:
        """
        Update strategy performance metrics

        Args:
            trades: List of completed trades
        """
        if not trades:
            return

        try:
            # Calculate basic metrics
            returns = [trade.get("return", 0) for trade in trades]

            self.performance.total_return = sum(returns)
            self.performance.total_trades = len(trades)

            winning_trades = [r for r in returns if r > 0]
            self.performance.win_rate = (
                len(winning_trades) / len(returns) if returns else 0
            )

            # Calculate Sharpe ratio (simplified)
            if len(returns) > 1:
                returns_std = np.std(returns)
                if returns_std > 0:
                    self.performance.sharpe_ratio = np.mean(returns) / returns_std

            # Calculate maximum drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            self.performance.max_drawdown = (
                abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            )

            # Calculate average trade duration
            durations = [
                trade.get("duration", 0) for trade in trades if "duration" in trade
            ]
            self.performance.avg_trade_duration = np.mean(durations) if durations else 0

            self.performance.last_updated = pd.Timestamp.now()

            logger.debug(f"{self.name}: Updated performance metrics")

        except Exception as e:
            logger.error(f"{self.name}: Error updating performance: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current strategy performance metrics

        Returns:
            Dictionary with performance metrics
        """
        return {
            "strategy_name": self.performance.strategy_name,
            "total_return": self.performance.total_return,
            "win_rate": self.performance.win_rate,
            "sharpe_ratio": self.performance.sharpe_ratio,
            "max_drawdown": self.performance.max_drawdown,
            "total_trades": self.performance.total_trades,
            "avg_trade_duration": self.performance.avg_trade_duration,
            "last_updated": self.performance.last_updated,
            "status": self.status.value,
        }

    def update_positions(self, symbol: str, current_price: float) -> None:
        """
        Update position P&L with current market price

        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol in self.positions:
            self.positions[symbol].update_pnl(current_price)

    def get_current_positions(self) -> Dict[str, Position]:
        """
        Get all current positions

        Returns:
            Dictionary of current positions by symbol
        """
        return self.positions.copy()

    def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """
        Close a position

        Args:
            symbol: Symbol to close position for
            reason: Reason for closing position

        Returns:
            True if position was closed successfully
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"{self.name}: Closed position for {symbol} - {reason}")
            return True
        return False

    def add_signal_to_history(self, signal: TradingSignal) -> None:
        """
        Add signal to historical record

        Args:
            signal: Trading signal to record
        """
        self.signals_history.append(signal)

        # Keep only last 1000 signals to prevent memory issues
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]

    def get_signals_history(self, hours: int = 24) -> List[TradingSignal]:
        """
        Get signals history for specified time period

        Args:
            hours: Number of hours to look back

        Returns:
            List of signals from specified time period
        """
        cutoff_time = pd.Timestamp.now() - timedelta(hours=hours)
        return [
            signal for signal in self.signals_history if signal.timestamp >= cutoff_time
        ]

    def set_status(self, status: StrategyStatus) -> None:
        """
        Set strategy status

        Args:
            status: New strategy status
        """
        old_status = self.status
        self.status = status
        logger.info(
            f"{self.name}: Status changed from {old_status.value} to {status.value}"
        )

    def is_active(self) -> bool:
        """
        Check if strategy is active

        Returns:
            True if strategy is active
        """
        return self.status == StrategyStatus.ACTIVE and self.config.enabled

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information

        Returns:
            Dictionary with strategy information
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "weight": self.config.weight,
            "risk_limit": self.config.risk_limit,
            "max_positions": self.config.max_positions,
            "current_positions": len(self.positions),
            "parameters": self.config.parameters,
            "symbols": self.config.symbols,
            "signals_count_24h": len(self.get_signals_history(24)),
            "performance": self.get_performance_metrics(),
        }

    def reset(self) -> None:
        """
        Reset strategy state (useful for backtesting)
        """
        self.positions.clear()
        self.signals_history.clear()
        self.performance = StrategyPerformance(strategy_name=self.name)
        self.risk_metrics = RiskMetrics()
        self._cache.clear()
        self.status = StrategyStatus.INACTIVE
        logger.info(f"{self.name}: Strategy state reset")

    def __repr__(self) -> str:
        """String representation of the strategy"""
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"

    # Enhanced Risk Management Integration
    async def validate_trade_with_risk_management(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> Dict[str, Any]:
        """
        Validate trade with integrated risk management

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price

        Returns:
            Validation result with risk assessment
        """
        result = {
            "approved": False,
            "position_size": 0,
            "reason": "",
            "risk_score": 0,
            "stop_loss": None,
            "take_profit": None,
        }

        try:
            # Basic signal validation
            if not self.validate_signal(signal):
                result["reason"] = "Invalid signal"
                return result

            # Risk manager trade allowance check
            if not self.risk_manager.check_trade_allowed(
                signal.symbol, signal.signal_type.value
            ):
                result["reason"] = "Risk manager blocked trade"
                return result

            # Calculate position size with risk consideration
            raw_position_size = self.get_position_size(
                signal, portfolio_value, current_price
            )

            # Apply risk-adjusted position sizing
            max_position_value = portfolio_value * self.config.risk_limit
            max_shares = max_position_value / current_price

            adjusted_position_size = min(abs(raw_position_size), max_shares)
            if raw_position_size < 0:
                adjusted_position_size = -adjusted_position_size

            # Calculate stop loss and take profit
            stop_loss_pct = 0.02  # 2% stop loss
            take_profit_pct = 0.04  # 4% take profit (2:1 reward/risk)

            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)

            # Calculate risk score
            position_risk = (
                abs(adjusted_position_size * current_price) / portfolio_value
            )
            signal_risk = 1 - signal.strength  # Lower strength = higher risk
            risk_score = int((position_risk + signal_risk) * 50)

            result.update(
                {
                    "approved": True,
                    "position_size": adjusted_position_size,
                    "reason": "Trade approved with risk management",
                    "risk_score": risk_score,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            )

        except Exception as e:
            logger.error(f"{self.name}: Error in risk validation: {e}")
            result["reason"] = f"Risk validation error: {e}"

        return result

    def record_trade(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_time: datetime,
        exit_time: datetime,
        reason: str = "Signal",
    ) -> None:
        """
        Record completed trade for performance tracking

        Args:
            symbol: Trading symbol
            signal_type: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            reason: Exit reason
        """
        # Calculate trade metrics
        if signal_type in ["BUY", "STRONG_BUY"]:
            pnl = (exit_price - entry_price) * quantity
            return_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) * quantity
            return_pct = (entry_price - exit_price) / entry_price

        trade_duration = (exit_time - entry_time).total_seconds() / 3600  # hours

        trade_record = {
            "strategy_name": self.name,
            "symbol": symbol,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "return": return_pct,
            "duration": trade_duration,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "exit_reason": reason,
        }

        self.trades_history.append(trade_record)

        # Update performance metrics
        self.update_performance([trade_record])

        logger.info(
            f"{self.name}: Trade recorded - {symbol} {signal_type} "
            f"PnL: ${pnl:.2f} ({return_pct:.2%})"
        )

    def calculate_advanced_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics

        Returns:
            Dictionary with advanced performance metrics
        """
        if not self.trades_history:
            return {}

        trades_df = pd.DataFrame(self.trades_history)
        returns = trades_df["return"].values

        metrics = {}

        # Basic metrics
        metrics["total_trades"] = len(returns)
        metrics["winning_trades"] = len(returns[returns > 0])
        metrics["losing_trades"] = len(returns[returns < 0])
        metrics["win_rate"] = (
            metrics["winning_trades"] / metrics["total_trades"]
            if metrics["total_trades"] > 0
            else 0
        )

        # Return metrics
        metrics["total_return"] = np.sum(returns)
        metrics["average_return"] = np.mean(returns)
        metrics["median_return"] = np.median(returns)

        # Risk metrics
        metrics["volatility"] = np.std(returns) if len(returns) > 1 else 0
        metrics["sharpe_ratio"] = (
            metrics["average_return"] / metrics["volatility"]
            if metrics["volatility"] > 0
            else 0
        )

        # Drawdown calculation
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        metrics["max_drawdown"] = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Average win/loss
        if metrics["winning_trades"] > 0:
            metrics["average_win"] = np.mean(returns[returns > 0])
        else:
            metrics["average_win"] = 0

        if metrics["losing_trades"] > 0:
            metrics["average_loss"] = np.mean(returns[returns < 0])
        else:
            metrics["average_loss"] = 0

        # Risk-reward ratio
        if metrics["average_loss"] != 0:
            metrics["risk_reward_ratio"] = abs(
                metrics["average_win"] / metrics["average_loss"]
            )
        else:
            metrics["risk_reward_ratio"] = float("inf")

        # Maximum consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for ret in returns:
            if ret > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif ret < 0:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                win_streak = 0
                loss_streak = 0

        metrics["max_consecutive_wins"] = max_win_streak
        metrics["max_consecutive_losses"] = max_loss_streak

        # Time-based metrics
        if "duration" in trades_df.columns:
            metrics["average_trade_duration"] = trades_df["duration"].mean()
            metrics["median_trade_duration"] = trades_df["duration"].median()

        return metrics

    def get_risk_report(self) -> Dict[str, Any]:
        """
        Get comprehensive risk report

        Returns:
            Risk report dictionary
        """
        # Get risk manager report
        risk_report = self.risk_manager.get_risk_report()

        # Add strategy-specific risk metrics
        performance_metrics = self.calculate_advanced_performance_metrics()

        # Calculate position exposure
        total_exposure = sum(
            abs(pos.size * pos.current_price) for pos in self.positions.values()
        )

        strategy_risk = {
            "strategy_name": self.name,
            "total_exposure": total_exposure,
            "exposure_ratio": total_exposure / self.initial_capital,
            "active_positions": len(self.positions),
            "performance_metrics": performance_metrics,
            "signals_last_24h": len(self.get_signals_history(24)),
        }

        risk_report["strategy_specific"] = strategy_risk

        return risk_report

    async def process_signals_with_risk_management(
        self,
        signals: List[TradingSignal],
        market_data: Dict[str, Dict],
        portfolio_value: float,
    ) -> List[Dict]:
        """
        Process signals through risk management pipeline

        Args:
            signals: List of trading signals
            market_data: Current market data
            portfolio_value: Current portfolio value

        Returns:
            List of approved trading actions
        """
        approved_actions = []

        for signal in signals:
            if signal.symbol not in market_data:
                logger.warning(f"{self.name}: No market data for {signal.symbol}")
                continue

            current_price = market_data[signal.symbol].get("price", signal.price)

            # Validate with risk management
            validation = await self.validate_trade_with_risk_management(
                signal, portfolio_value, current_price
            )

            if validation["approved"]:
                action = {
                    "signal": signal,
                    "action_type": "OPEN_POSITION",
                    "symbol": signal.symbol,
                    "quantity": validation["position_size"],
                    "price": current_price,
                    "stop_loss": validation["stop_loss"],
                    "take_profit": validation["take_profit"],
                    "risk_score": validation["risk_score"],
                    "strategy_name": self.name,
                }
                approved_actions.append(action)

                # Add signal to history
                self.add_signal_to_history(signal)
            else:
                logger.info(
                    f"{self.name}: Signal rejected for {signal.symbol}: {validation['reason']}"
                )

        return approved_actions

    def update_portfolio_value(self, current_value: float) -> None:
        """
        Update portfolio value and calculate daily returns

        Args:
            current_value: Current portfolio value
        """
        if len(self.portfolio_values) > 0:
            previous_value = self.portfolio_values[-1]
            daily_return = (current_value - previous_value) / previous_value
            self.daily_returns.append(daily_return)

        self.portfolio_values.append(current_value)

        # Update risk manager with daily P&L
        if len(self.portfolio_values) > 1:
            daily_pnl = current_value - self.portfolio_values[-2]
            self.risk_manager.daily_pnl = daily_pnl
