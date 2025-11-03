# quant_project/execution/portfolio.py
# Portfolio management for live trading

import logging
from core.event import SignalEvent, FillEvent, OrderEvent
from core.event_loop import EventLoop

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, event_queue: EventLoop):
        self.event_queue = event_queue
        self.positions = {}  # {symbol: quantity}
        self.holdings = {}  # {symbol: market_value}
        self.cash = 100000  # Default initial capital
        self.initial_capital = self.cash

        logger.info(f"Portfolio initialized with capital: {self.cash}")

    async def on_signal(self, signal: SignalEvent):
        """Handle signal event and generate order"""
        logger.info(
            f"Portfolio received signal: {signal.symbol} {signal.direction} {signal.quantity}"
        )

        # Simple order generation logic
        if signal.direction in ["BUY", "SELL"]:
            order = OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                direction=signal.direction,
                quantity=signal.quantity,
            )
            await self.event_queue.put_event(order)
            logger.info(
                f"Order generated: {order.symbol} {order.direction} {order.quantity}"
            )

    async def on_fill(self, fill: FillEvent):
        """Handle fill event and update positions"""
        logger.info(
            f"Portfolio received fill: {fill.symbol} {fill.direction} {fill.quantity} @ {fill.fill_price}"
        )

        # Update positions
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = 0

        if fill.direction == "BUY":
            self.positions[fill.symbol] += fill.quantity
            self.cash -= fill.quantity * fill.fill_price + fill.commission
        elif fill.direction == "SELL":
            self.positions[fill.symbol] -= fill.quantity
            self.cash += fill.quantity * fill.fill_price - fill.commission

        # Update holdings
        self.holdings[fill.symbol] = self.positions[fill.symbol] * fill.fill_price

        # Calculate total portfolio value
        total_value = self.cash + sum(self.holdings.values())
        pnl = total_value - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100

        logger.info(
            f"Portfolio Update - Cash: {self.cash:.2f}, Total Value: {total_value:.2f}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)"
        )
        logger.info(f"Current Positions: {self.positions}")

    def get_current_positions(self):
        """Get current positions"""
        return self.positions.copy()

    def get_portfolio_value(self):
        """Get total portfolio value"""
        return self.cash + sum(self.holdings.values())
