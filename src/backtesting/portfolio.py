"""
Portfolio management for backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import logging


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a position in the portfolio
    """
    instrument: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L"""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        """Return percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    def update_price(self, new_price: float) -> None:
        """Update current price"""
        self.current_price = new_price
    
    def add_transaction(self, quantity: float, price: float, commission: float = 0.0) -> None:
        """
        Add to position (positive quantity) or reduce position (negative quantity)
        
        Args:
            quantity: Quantity to add/remove
            price: Transaction price
            commission: Transaction commission
        """
        if quantity == 0:
            return
            
        # If reducing position, calculate realized P&L
        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            # Calculate realized P&L for the closed portion
            closed_quantity = min(abs(quantity), abs(self.quantity))
            if self.quantity > 0:
                self.realized_pnl += closed_quantity * (price - self.entry_price)
            else:
                self.realized_pnl += closed_quantity * (self.entry_price - price)
        
        # Update position
        new_quantity = self.quantity + quantity
        
        # If position changes side or is new, update entry price
        if (self.quantity >= 0 and new_quantity < 0) or (self.quantity <= 0 and new_quantity > 0):
            self.entry_price = price
        elif abs(new_quantity) > abs(self.quantity):
            # Position increased, calculate weighted average entry price
            if new_quantity != 0:
                self.entry_price = (
                    (self.entry_price * abs(self.quantity) + price * abs(quantity)) /
                    abs(new_quantity)
                )
        
        self.quantity = new_quantity
        self.commission_paid += commission
        self.current_price = price


class Portfolio:
    """
    Portfolio manager for backtesting
    """
    
    def __init__(
        self,
        initial_capital: float,
        currency: str = 'USD',
        track_history: bool = True
    ):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital
            currency: Base currency
            track_history: Whether to track portfolio history
        """
        self.initial_capital = initial_capital
        self.currency = currency
        self.track_history = track_history
        
        # Current state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # History tracking
        self.history = []
        self.transaction_history = []
        self.current_datetime = None
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def total_exposure(self) -> float:
        """Total market exposure"""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    @property
    def exposure_pct(self) -> float:
        """Exposure as percentage of total value"""
        if self.total_value == 0:
            return 0.0
        return self.total_exposure / self.total_value
    
    @property
    def position_count(self) -> int:
        """Number of open positions"""
        return len([p for p in self.positions.values() if p.quantity != 0])
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L"""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total P&L"""
        return self.total_value - self.initial_capital
    
    @property
    def return_pct(self) -> float:
        """Total return percentage"""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    def update_market_prices(
        self,
        prices: Dict[str, float],
        datetime: Optional[datetime] = None
    ) -> None:
        """
        Update market prices for all positions
        
        Args:
            prices: Dictionary of instrument prices
            datetime: Current datetime
        """
        self.current_datetime = datetime or self.current_datetime
        
        for instrument, position in self.positions.items():
            if instrument in prices:
                position.update_price(prices[instrument])
        
        # Record history
        if self.track_history:
            self._record_history()
    
    def execute_order(
        self,
        instrument: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        order_type: str = 'MARKET',
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Execute an order
        
        Args:
            instrument: Instrument to trade
            quantity: Order quantity (positive for buy, negative for sell)
            price: Execution price (before slippage)
            commission: Transaction commission
            slippage: Price slippage
            order_type: Order type
            metadata: Additional order metadata
            
        Returns:
            True if order executed successfully
        """
        if quantity == 0:
            return False
        
        # Calculate execution price with slippage
        execution_price = price + slippage
        
        # Calculate total cost
        trade_value = abs(quantity * execution_price)
        total_cost = trade_value + commission
        
        # Check if we have enough cash for buy orders
        if quantity > 0 and total_cost > self.cash:
            logger.warning(
                f"Insufficient cash for order: need {total_cost:.2f}, "
                f"have {self.cash:.2f}"
            )
            return False
        
        # Update or create position
        if instrument not in self.positions:
            self.positions[instrument] = Position(
                instrument=instrument,
                quantity=0,
                entry_price=execution_price,
                entry_time=self.current_datetime,
                current_price=execution_price,
                metadata=metadata or {}
            )
        
        position = self.positions[instrument]
        position.add_transaction(quantity, execution_price, commission)
        
        # Update cash
        self.cash -= quantity * execution_price + commission
        
        # Update tracking
        self.total_commission += commission
        self.total_slippage += abs(slippage * quantity)
        self.total_trades += 1
        
        # Record transaction
        self._record_transaction(
            instrument, quantity, price, execution_price,
            commission, slippage, order_type, metadata
        )
        
        # Remove position if closed
        if position.quantity == 0:
            del self.positions[instrument]
        
        return True
    
    def get_position(self, instrument: str) -> Optional[Position]:
        """Get position for an instrument"""
        return self.positions.get(instrument)
    
    def get_position_size(self, instrument: str) -> float:
        """Get position size for an instrument"""
        position = self.positions.get(instrument)
        return position.quantity if position else 0.0
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        total_value = self.total_value
        if total_value == 0:
            return {}
        
        weights = {}
        for instrument, position in self.positions.items():
            weights[instrument] = position.market_value / total_value
        
        weights['cash'] = self.cash / total_value
        return weights
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return {
            'datetime': self.current_datetime,
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': {
                inst: {
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'return_pct': pos.return_pct,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'holding_period': (
                        (self.current_datetime - pos.entry_time).days
                        if self.current_datetime and pos.entry_time else 0
                    )
                }
                for inst, pos in self.positions.items()
            },
            'total_exposure': self.total_exposure,
            'exposure_pct': self.exposure_pct,
            'position_count': self.position_count,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'return_pct': self.return_pct,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_trades': self.total_trades
        }
    
    def _record_history(self) -> None:
        """Record current portfolio state to history"""
        if not self.track_history:
            return
        
        state = self.get_portfolio_state()
        self.history.append(state)
    
    def _record_transaction(
        self,
        instrument: str,
        quantity: float,
        order_price: float,
        execution_price: float,
        commission: float,
        slippage: float,
        order_type: str,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Record transaction details"""
        transaction = {
            'datetime': self.current_datetime,
            'instrument': instrument,
            'quantity': quantity,
            'side': 'BUY' if quantity > 0 else 'SELL',
            'order_price': order_price,
            'execution_price': execution_price,
            'commission': commission,
            'slippage': slippage,
            'slippage_cost': slippage * abs(quantity),
            'order_type': order_type,
            'trade_value': abs(quantity * execution_price),
            'metadata': metadata or {}
        }
        
        self.transaction_history.append(transaction)
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        if not self.transaction_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.transaction_history)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        if not self.history:
            return pd.DataFrame()
        
        # Extract time series data
        data = {
            'datetime': [h['datetime'] for h in self.history],
            'total_value': [h['total_value'] for h in self.history],
            'cash': [h['cash'] for h in self.history],
            'total_exposure': [h['total_exposure'] for h in self.history],
            'position_count': [h['position_count'] for h in self.history],
            'unrealized_pnl': [h['unrealized_pnl'] for h in self.history],
            'realized_pnl': [h['realized_pnl'] for h in self.history],
            'total_pnl': [h['total_pnl'] for h in self.history],
            'return_pct': [h['return_pct'] for h in self.history]
        }
        
        df = pd.DataFrame(data)
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        
        return df
    
    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.history.clear()
        self.transaction_history.clear()
        self.current_datetime = None
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0