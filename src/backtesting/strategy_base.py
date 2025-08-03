"""
Base strategy class for backtesting
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class Signal:
    """
    Trading signal class
    """
    
    def __init__(
        self,
        instrument: str,
        direction: str,  # 'BUY', 'SELL', 'HOLD'
        strength: float = 1.0,  # Signal strength [0, 1]
        target_weight: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trading signal
        
        Args:
            instrument: Instrument to trade
            direction: Trade direction
            strength: Signal strength (0-1)
            target_weight: Target portfolio weight
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional signal metadata
        """
        self.instrument = instrument
        self.direction = direction.upper()
        self.strength = max(0.0, min(1.0, strength))
        self.target_weight = target_weight
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def __repr__(self):
        return f"Signal({self.instrument}, {self.direction}, strength={self.strength:.2f})"


class Strategy(ABC):
    """
    Abstract base class for trading strategies
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            risk_parameters: Risk management parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.risk_parameters = risk_parameters or self._default_risk_parameters()
        
        # Internal state
        self._initialized = False
        self._current_signals = {}
        self._signal_history = []
        self._performance_metrics = {}
        
    def _default_risk_parameters(self) -> Dict[str, Any]:
        """Get default risk parameters"""
        return {
            'max_position_size': 0.1,  # Max 10% per position
            'max_portfolio_exposure': 1.0,  # Max 100% invested
            'stop_loss_pct': 0.02,  # 2% stop loss
            'position_sizing_method': 'fixed',  # fixed, volatility, kelly
            'max_positions': 10,  # Maximum number of positions
            'min_holding_period': 0,  # Minimum bars to hold
            'use_stop_loss': True,
            'use_take_profit': False,
            'take_profit_pct': 0.05  # 5% take profit
        }
    
    @abstractmethod
    def initialize(self, market_data: pd.DataFrame) -> None:
        """
        Initialize strategy with historical data
        
        Args:
            market_data: Historical market data for initialization
        """
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate trading signals based on market data
        
        Args:
            market_data: Current market data
            portfolio_state: Current portfolio state
            
        Returns:
            List of trading signals
        """
        pass
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate position size based on signal and risk parameters
        
        Args:
            signal: Trading signal
            current_price: Current price of the instrument
            portfolio_value: Total portfolio value
            current_positions: Current positions
            
        Returns:
            Position size (number of units)
        """
        method = self.risk_parameters.get('position_sizing_method', 'fixed')
        
        if method == 'fixed':
            # Fixed percentage of portfolio
            max_position = self.risk_parameters.get('max_position_size', 0.1)
            position_value = portfolio_value * max_position * signal.strength
            return int(position_value / current_price)
            
        elif method == 'volatility':
            # Inverse volatility sizing
            # Requires volatility data in signal metadata
            volatility = signal.metadata.get('volatility', 0.02)
            target_risk = self.risk_parameters.get('target_risk', 0.01)
            position_value = (portfolio_value * target_risk) / volatility
            return int(position_value / current_price)
            
        elif method == 'kelly':
            # Kelly criterion sizing
            # Requires win probability and win/loss ratio in metadata
            win_prob = signal.metadata.get('win_probability', 0.5)
            win_loss_ratio = signal.metadata.get('win_loss_ratio', 1.0)
            
            # Kelly formula: f = p - q/b
            # where p = win probability, q = loss probability, b = win/loss ratio
            kelly_fraction = win_prob - (1 - win_prob) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_value = portfolio_value * kelly_fraction * signal.strength
            return int(position_value / current_price)
            
        else:
            # Default to fixed sizing
            return int(portfolio_value * 0.1 * signal.strength / current_price)
    
    def apply_risk_filters(
        self,
        signals: List[Signal],
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Apply risk management filters to signals
        
        Args:
            signals: List of raw signals
            portfolio_state: Current portfolio state
            
        Returns:
            Filtered signals that pass risk checks
        """
        if not portfolio_state:
            return signals
            
        filtered_signals = []
        
        # Check maximum positions
        current_positions = portfolio_state.get('positions', {})
        max_positions = self.risk_parameters.get('max_positions', 10)
        
        # Check portfolio exposure
        current_exposure = portfolio_state.get('total_exposure', 0)
        max_exposure = self.risk_parameters.get('max_portfolio_exposure', 1.0)
        
        for signal in signals:
            # Skip if already at max positions (unless closing)
            if len(current_positions) >= max_positions and signal.direction != 'SELL':
                logger.debug(f"Skipping {signal.instrument}: max positions reached")
                continue
                
            # Skip if would exceed max exposure
            if signal.direction == 'BUY' and current_exposure >= max_exposure:
                logger.debug(f"Skipping {signal.instrument}: max exposure reached")
                continue
                
            # Check minimum holding period
            if signal.instrument in current_positions:
                position = current_positions[signal.instrument]
                holding_period = position.get('holding_period', 0)
                min_period = self.risk_parameters.get('min_holding_period', 0)
                
                if holding_period < min_period:
                    logger.debug(
                        f"Skipping {signal.instrument}: "
                        f"holding period {holding_period} < {min_period}"
                    )
                    continue
            
            # Add stop loss and take profit if enabled
            if self.risk_parameters.get('use_stop_loss', True) and not signal.stop_loss:
                if signal.direction == 'BUY':
                    current_price = signal.metadata.get('current_price', 0)
                    stop_loss_pct = self.risk_parameters.get('stop_loss_pct', 0.02)
                    signal.stop_loss = current_price * (1 - stop_loss_pct)
                    
            if self.risk_parameters.get('use_take_profit', False) and not signal.take_profit:
                if signal.direction == 'BUY':
                    current_price = signal.metadata.get('current_price', 0)
                    take_profit_pct = self.risk_parameters.get('take_profit_pct', 0.05)
                    signal.take_profit = current_price * (1 + take_profit_pct)
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """
        Update strategy performance metrics
        
        Args:
            metrics: Performance metrics dictionary
        """
        self._performance_metrics.update(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary
        
        Returns:
            Performance metrics dictionary
        """
        return {
            'strategy_name': self.name,
            'parameters': self.parameters,
            'risk_parameters': self.risk_parameters,
            'performance_metrics': self._performance_metrics,
            'total_signals': len(self._signal_history),
            'active_signals': len(self._current_signals)
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self._initialized = False
        self._current_signals = {}
        self._signal_history = []
        self._performance_metrics = {}
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy and hold strategy for benchmarking
    """
    
    def __init__(self, instruments: List[str], weights: Optional[List[float]] = None):
        """
        Initialize buy and hold strategy
        
        Args:
            instruments: List of instruments to hold
            weights: Portfolio weights (equal weight if None)
        """
        super().__init__(name="BuyAndHold")
        self.instruments = instruments
        
        if weights is None:
            self.weights = [1.0 / len(instruments)] * len(instruments)
        else:
            self.weights = weights
            
        self._initial_signal_sent = False
    
    def initialize(self, market_data: pd.DataFrame) -> None:
        """Initialize strategy"""
        self._initialized = True
        self._initial_signal_sent = False
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """Generate buy signals on first call, then hold"""
        signals = []
        
        # Only send buy signals once at the beginning
        if not self._initial_signal_sent:
            for instrument, weight in zip(self.instruments, self.weights):
                signal = Signal(
                    instrument=instrument,
                    direction='BUY',
                    strength=1.0,
                    target_weight=weight,
                    metadata={'strategy': 'BuyAndHold'}
                )
                signals.append(signal)
            
            self._initial_signal_sent = True
            self._signal_history.extend(signals)
        
        return signals