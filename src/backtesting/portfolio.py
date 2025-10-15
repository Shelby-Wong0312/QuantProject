"""
Portfolio Management - Handles position tracking, P&L calculation, and trade execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, new_price: float) -> float:
        """Update current price and calculate unrealized P&L"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        return self.unrealized_pnl

    def get_market_value(self) -> float:
        """Get current market value of position"""
        return abs(self.quantity) * self.current_price

    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0

    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0


@dataclass
class Trade:
    """Represents a completed trade"""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    slippage: float
    duration_hours: float
    signal_type: str
    exit_reason: str = "Signal"


class Portfolio:
    """
    Portfolio management system for backtesting

    Features:
    - Position tracking
    - P&L calculation
    - Transaction cost modeling
    - Trade execution simulation
    """

    def __init__(
        self, initial_capital: float = 100000.0, commission: float = 0.001, slippage: float = 0.0005
    ):
        """
        Initialize portfolio

        Args:
            initial_capital: Starting cash amount
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission
        self.slippage_rate = slippage

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []

        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.total_trades = 0

        # Portfolio history
        self.value_history: List[Tuple[datetime, float]] = []

        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")

    def execute_trade(
        self, symbol: str, quantity: float, price: float, signal_type: str, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Execute a trade (buy/sell)

        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive=buy, negative=sell)
            price: Execution price
            signal_type: Type of signal triggering trade
            timestamp: Trade timestamp

        Returns:
            Trade execution result
        """
        try:
            # Calculate transaction costs
            trade_value = abs(quantity * price)
            commission = trade_value * self.commission_rate
            slippage_cost = trade_value * self.slippage_rate
            total_cost = trade_value + commission + slippage_cost

            # Apply slippage to price
            if quantity > 0:  # Buying
                execution_price = price * (1 + self.slippage_rate)
            else:  # Selling
                execution_price = price * (1 - self.slippage_rate)

            result = {
                "success": False,
                "symbol": symbol,
                "quantity": quantity,
                "price": execution_price,
                "commission": commission,
                "slippage": slippage_cost,
                "timestamp": timestamp,
                "signal_type": signal_type,
            }

            if quantity > 0:  # Buy order
                result.update(
                    self._execute_buy(
                        symbol, quantity, execution_price, commission, timestamp, signal_type
                    )
                )
            else:  # Sell order
                result.update(
                    self._execute_sell(
                        symbol, abs(quantity), execution_price, commission, timestamp, signal_type
                    )
                )

            if result["success"]:
                self.total_commission_paid += commission
                self.total_slippage_cost += slippage_cost
                self.total_trades += 1

            return result

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
            }

    def _execute_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime,
        signal_type: str,
    ) -> Dict:
        """Execute buy order"""
        total_cost = quantity * price + commission

        # Check if we have enough cash
        if self.cash < total_cost:
            logger.warning(
                f"Insufficient cash for {symbol}: need ${total_cost:.2f}, have ${self.cash:.2f}"
            )
            return {
                "success": False,
                "reason": "Insufficient cash",
                "required": total_cost,
                "available": self.cash,
            }

        # Update cash
        self.cash -= total_cost

        # Add to position
        if symbol in self.positions:
            # Average down existing position
            existing_pos = self.positions[symbol]
            total_quantity = existing_pos.quantity + quantity
            total_value = existing_pos.quantity * existing_pos.entry_price + quantity * price
            avg_price = total_value / total_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                entry_price=avg_price,
                current_price=price,
                entry_time=existing_pos.entry_time,
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                entry_time=timestamp,
            )

        logger.debug(f"Bought {quantity:.0f} shares of {symbol} @ ${price:.2f}")

        return {
            "success": True,
            "action": "BUY",
            "total_cost": total_cost,
            "remaining_cash": self.cash,
        }

    def _execute_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime,
        signal_type: str,
    ) -> Dict:
        """Execute sell order"""
        # Check if we have enough shares
        if symbol not in self.positions:
            logger.warning(f"No position to sell for {symbol}")
            return {"success": False, "reason": "No position to sell"}

        position = self.positions[symbol]

        if position.quantity < quantity:
            logger.warning(
                f"Insufficient shares for {symbol}: need {quantity}, have {position.quantity}"
            )
            # Partial sale - sell what we have
            quantity = position.quantity

        # Calculate proceeds
        proceeds = quantity * price - commission

        # Calculate realized P&L
        realized_pnl = (price - position.entry_price) * quantity

        # Update cash
        self.cash += proceeds

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=quantity,
            pnl=realized_pnl,
            commission=commission,
            slippage=quantity * price * self.slippage_rate,
            duration_hours=(timestamp - position.entry_time).total_seconds() / 3600,
            signal_type=signal_type,
        )

        self.closed_trades.append(trade)

        # Update or close position
        if position.quantity == quantity:
            # Close entire position
            del self.positions[symbol]
            logger.debug(f"Closed position: {symbol} - P&L: ${realized_pnl:.2f}")
        else:
            # Partial sale
            position.quantity -= quantity
            position.unrealized_pnl = (price - position.entry_price) * position.quantity
            logger.debug(
                f"Partial sale: {symbol} - Sold {quantity}, Remaining: {position.quantity}"
            )

        return {
            "success": True,
            "action": "SELL",
            "proceeds": proceeds,
            "realized_pnl": realized_pnl,
            "remaining_cash": self.cash,
        }

    def update_position_price(self, symbol: str, new_price: float) -> None:
        """Update position with current market price"""
        if symbol in self.positions:
            self.positions[symbol].update_price(new_price)

    def calculate_total_value(self, market_data: Optional[Dict[str, Dict]] = None) -> float:
        """
        Calculate total portfolio value

        Args:
            market_data: Current market prices {symbol: {'close': price}}

        Returns:
            Total portfolio value
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            if market_data and symbol in market_data:
                # Use provided market data
                current_price = market_data[symbol]["close"]
                position.update_price(current_price)

            # Add position value
            total_value += position.get_market_value()

        return total_value

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_realized_pnl(self) -> float:
        """Get total realized P&L from closed trades"""
        return sum(trade.pnl for trade in self.closed_trades)

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_positions_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions"""
        summary = {}

        for symbol, position in self.positions.items():
            summary[symbol] = {
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "market_value": position.get_market_value(),
                "unrealized_pnl": position.unrealized_pnl,
                "return_pct": (
                    (position.current_price - position.entry_price) / position.entry_price
                )
                * 100,
                "days_held": (datetime.now() - position.entry_time).days,
                "is_long": position.is_long(),
            }

        return summary

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_value = self.calculate_total_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital

        # Position statistics
        long_positions = sum(1 for pos in self.positions.values() if pos.is_long())
        short_positions = sum(1 for pos in self.positions.values() if pos.is_short())

        # Calculate exposure
        long_exposure = sum(
            pos.get_market_value() for pos in self.positions.values() if pos.is_long()
        )
        short_exposure = sum(
            pos.get_market_value() for pos in self.positions.values() if pos.is_short()
        )

        return {
            "initial_capital": self.initial_capital,
            "current_cash": self.cash,
            "total_value": total_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "unrealized_pnl": self.get_unrealized_pnl(),
            "realized_pnl": self.get_realized_pnl(),
            "total_pnl": self.get_total_pnl(),
            "num_positions": len(self.positions),
            "long_positions": long_positions,
            "short_positions": short_positions,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "total_trades": self.total_trades,
            "total_commission": self.total_commission_paid,
            "total_slippage": self.total_slippage_cost,
        }

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get trading statistics from closed trades"""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_duration_hours": 0,
            }

        pnls = [trade.pnl for trade in self.closed_trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        total_trades = len(self.closed_trades)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        durations = [trade.duration_hours for trade in self.closed_trades]

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "largest_win": max(pnls) if pnls else 0,
            "largest_loss": min(pnls) if pnls else 0,
            "avg_duration_hours": sum(durations) / len(durations) if durations else 0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": gross_profit + gross_loss,  # gross_loss is negative
        }

    def close_all_positions(
        self, market_data: Dict[str, Dict], timestamp: datetime, reason: str = "Close all"
    ) -> List[Dict]:
        """
        Close all open positions

        Args:
            market_data: Current market prices
            timestamp: Closing timestamp
            reason: Reason for closing

        Returns:
            List of closing trade results
        """
        closing_trades = []
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            if symbol in market_data:
                position = self.positions[symbol]
                current_price = market_data[symbol]["close"]

                # Execute sell order
                result = self.execute_trade(
                    symbol=symbol,
                    quantity=-position.quantity,  # Sell entire position
                    price=current_price,
                    signal_type="CLOSE",
                    timestamp=timestamp,
                )

                if result["success"]:
                    closing_trades.append(result)
                    logger.info(f"Closed position: {symbol} @ ${current_price:.2f} - {reason}")

        return closing_trades

    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.total_trades = 0
        self.value_history.clear()

        logger.info("Portfolio reset to initial state")

    def get_exposure_by_symbol(self) -> Dict[str, float]:
        """Get exposure percentage by symbol"""
        total_value = self.calculate_total_value()
        if total_value == 0:
            return {}

        exposure = {}
        for symbol, position in self.positions.items():
            exposure[symbol] = (position.get_market_value() / total_value) * 100

        return exposure

    @property
    def total_value(self) -> float:
        """Get current total portfolio value"""
        return self.calculate_total_value()

    def __repr__(self) -> str:
        """String representation of portfolio"""
        total_value = self.calculate_total_value()
        return (
            f"Portfolio(value=${total_value:,.2f}, cash=${self.cash:,.2f}, "
            f"positions={len(self.positions)}, trades={self.total_trades})"
        )
