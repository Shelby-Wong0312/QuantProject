"""
Base Strategy Class - Abstract base class for all trading strategies
Provides standardized interface and common functionality for trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from .strategy_interface import (
    TradingSignal, SignalType, StrategyConfig, StrategyPerformance, 
    Position, RiskMetrics, StrategyStatus
)

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
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.name = config.name
        self.status = StrategyStatus.INACTIVE
        
        # Initialize strategy state
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.performance = StrategyPerformance(strategy_name=self.name)
        self.risk_metrics = RiskMetrics()
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Cache for performance optimization
        self._cache = {}
        
        logger.info(f"Initialized strategy: {self.name}")
    
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
    def get_position_size(self, signal: TradingSignal, portfolio_value: float, 
                         current_price: float) -> float:
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
    def apply_risk_management(self, position: Position, 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
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
            returns = [trade.get('return', 0) for trade in trades]
            
            self.performance.total_return = sum(returns)
            self.performance.total_trades = len(trades)
            
            winning_trades = [r for r in returns if r > 0]
            self.performance.win_rate = len(winning_trades) / len(returns) if returns else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(returns) > 1:
                returns_std = np.std(returns)
                if returns_std > 0:
                    self.performance.sharpe_ratio = np.mean(returns) / returns_std
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            self.performance.max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Calculate average trade duration
            durations = [trade.get('duration', 0) for trade in trades if 'duration' in trade]
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
            'strategy_name': self.performance.strategy_name,
            'total_return': self.performance.total_return,
            'win_rate': self.performance.win_rate,
            'sharpe_ratio': self.performance.sharpe_ratio,
            'max_drawdown': self.performance.max_drawdown,
            'total_trades': self.performance.total_trades,
            'avg_trade_duration': self.performance.avg_trade_duration,
            'last_updated': self.performance.last_updated,
            'status': self.status.value
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
            signal for signal in self.signals_history 
            if signal.timestamp >= cutoff_time
        ]
    
    def set_status(self, status: StrategyStatus) -> None:
        """
        Set strategy status
        
        Args:
            status: New strategy status
        """
        old_status = self.status
        self.status = status
        logger.info(f"{self.name}: Status changed from {old_status.value} to {status.value}")
    
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
            'name': self.name,
            'status': self.status.value,
            'enabled': self.config.enabled,
            'weight': self.config.weight,
            'risk_limit': self.config.risk_limit,
            'max_positions': self.config.max_positions,
            'current_positions': len(self.positions),
            'parameters': self.config.parameters,
            'symbols': self.config.symbols,
            'signals_count_24h': len(self.get_signals_history(24)),
            'performance': self.get_performance_metrics()
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