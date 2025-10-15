"""
Live Trading System - Main orchestrator for real-time trading
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)


class LiveTradingSystem:
    """Main live trading system orchestrator"""

    def __init__(self, config_path: str = "config/live_trading_config.json"):
        """Initialize live trading system"""
        self.config = self._load_config(config_path)
        self.strategies = {}
        self.symbols = []
        self.capital = 10000
        self.positions = {}
        self.orders = []
        self.is_running = False
        self.emergency_stop = False
        self.daily_pnl = 0
        self.max_daily_loss = -0.05  # 5% max daily loss

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
        return {
            "max_positions": 10,
            "position_size_pct": 0.1,
            "max_single_position": 0.2,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "daily_loss_limit": 0.05,
        }

    async def start_trading(self, strategies: List, symbols: List[str], capital: float = 10000):
        """Start live trading system"""
        logger.info(f"Starting live trading with capital: ${capital:,.2f}")

        self.strategies = strategies
        self.symbols = symbols
        self.capital = capital
        self.is_running = True

        # Initialize components
        await self._initialize_components()

        # Start trading loops
        tasks = [
            asyncio.create_task(self._data_feed_loop()),
            asyncio.create_task(self._strategy_loop()),
            asyncio.create_task(self._risk_monitor_loop()),
            asyncio.create_task(self._order_execution_loop()),
            asyncio.create_task(self._performance_tracking_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Trading system error: {e}")
            await self.emergency_shutdown()

    async def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing trading components...")

        # Initialize data feeds
        from data_pipeline.free_data_client import FreeDataClient

        self.data_client = FreeDataClient()

        # Initialize order manager
        from .order_manager import OrderManager

        self.order_manager = OrderManager()

        # Initialize risk monitor
        from .risk_monitor import RiskMonitor

        self.risk_monitor = RiskMonitor(self.config)

        logger.info("All components initialized successfully")

    async def _data_feed_loop(self):
        """Real-time data feed loop"""
        while self.is_running and not self.emergency_stop:
            try:
                # Get real-time quotes
                for symbol in self.symbols:
                    price = self.data_client.get_real_time_price(symbol)
                    if price:
                        await self._process_price_update(symbol, price)

                await asyncio.sleep(1)  # 1 second update interval

            except Exception as e:
                logger.error(f"Data feed error: {e}")
                await asyncio.sleep(5)

    async def _strategy_loop(self):
        """Strategy signal generation loop"""
        while self.is_running and not self.emergency_stop:
            try:
                for strategy in self.strategies:
                    for symbol in self.symbols:
                        # Get historical data
                        self.data_client.get_historical_data(symbol, period="1mo")
                        if data is not None and not data.empty:
                            # Generate signals
                            signal = strategy.generate_signals(data)
                            if signal != 0:
                                await self._process_signal(symbol, signal, strategy)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Strategy error: {e}")
                await asyncio.sleep(60)

    async def _risk_monitor_loop(self):
        """Real-time risk monitoring loop"""
        while self.is_running and not self.emergency_stop:
            try:
                # Check daily loss limit
                if self.daily_pnl / self.capital < self.max_daily_loss:
                    logger.critical(f"Daily loss limit reached: {self.daily_pnl/self.capital:.2%}")
                    await self.emergency_shutdown()
                    break

                # Check position limits
                for symbol, position in self.positions.items():
                    if abs(position["value"]) / self.capital > self.config["max_single_position"]:
                        logger.warning(f"Position limit exceeded for {symbol}")
                        await self._reduce_position(symbol)

                # Check stop losses
                await self.risk_monitor.check_stop_losses(self.positions)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(5)

    async def _order_execution_loop(self):
        """Order execution and management loop"""
        while self.is_running and not self.emergency_stop:
            try:
                # Process pending orders
                pending_orders = self.order_manager.get_pending_orders()
                for order in pending_orders:
                    result = await self.order_manager.execute_order(order)
                    if result["status"] == "filled":
                        await self._update_position(order)

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Order execution error: {e}")
                await asyncio.sleep(5)

    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        while self.is_running and not self.emergency_stop:
            try:
                # Calculate current P&L
                current_pnl = self._calculate_pnl()
                self.daily_pnl = current_pnl

                # Log performance
                logger.info(f"Current P&L: ${current_pnl:,.2f} ({current_pnl/self.capital:.2%})")
                logger.info(f"Positions: {len(self.positions)}, Orders: {len(self.orders)}")

                # Save to database
                self._save_performance_metrics()

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(30)

    async def _process_price_update(self, symbol: str, price: float):
        """Process real-time price update"""
        if symbol in self.positions:
            self.positions[symbol]["current_price"] = price
            self.positions[symbol]["unrealized_pnl"] = (
                price - self.positions[symbol]["entry_price"]
            ) * self.positions[symbol]["quantity"]

    async def _process_signal(self, symbol: str, signal: int, strategy):
        """Process trading signal from strategy"""
        # Check if we can take new position
        if not self.risk_monitor.can_take_position(self.positions, self.capital):
            logger.warning(f"Cannot take new position for {symbol} - risk limits")
            return

        # Calculate position size
        position_size = self._calculate_position_size(symbol, signal)

        # Create order
        order = {
            "symbol": symbol,
            "side": "buy" if signal > 0 else "sell",
            "quantity": position_size,
            "type": "market",
            "strategy": strategy.__class__.__name__,
            "timestamp": datetime.now(),
        }

        # Submit order
        await self.order_manager.submit_order(order)
        logger.info(f"Order submitted: {order}")

    def _calculate_position_size(self, symbol: str, signal: float) -> int:
        """Calculate position size based on risk management rules"""
        available_capital = self.capital * self.config["position_size_pct"]
        price = self.data_client.get_real_time_price(symbol)

        if price:
            shares = int(available_capital / price)
            return min(shares, 1000)  # Max 1000 shares per position
        return 0

    async def _update_position(self, order: Dict):
        """Update position after order execution"""
        symbol = order["symbol"]

        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0, "entry_price": 0, "value": 0}

        # Update position
        if order["side"] == "buy":
            self.positions[symbol]["quantity"] += order["quantity"]
        else:
            self.positions[symbol]["quantity"] -= order["quantity"]

        self.positions[symbol]["entry_price"] = order["price"]
        self.positions[symbol]["value"] = self.positions[symbol]["quantity"] * order["price"]

    async def _reduce_position(self, symbol: str):
        """Reduce position size to comply with limits"""
        if symbol in self.positions:
            # Reduce by 50%
            reduce_qty = self.positions[symbol]["quantity"] // 2
            order = {
                "symbol": symbol,
                "side": "sell" if self.positions[symbol]["quantity"] > 0 else "buy",
                "quantity": abs(reduce_qty),
                "type": "market",
                "reason": "risk_reduction",
            }
            await self.order_manager.submit_order(order)

    def _calculate_pnl(self) -> float:
        """Calculate total P&L"""
        total_pnl = 0
        for symbol, position in self.positions.items():
            if "unrealized_pnl" in position:
                total_pnl += position["unrealized_pnl"]
        return total_pnl

    def _save_performance_metrics(self):
        """Save performance metrics to file/database"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "positions": len(self.positions),
            "orders": len(self.orders),
        }

        # Save to JSON file
        metrics_file = Path("data/performance_metrics.json")
        metrics_file.parent.mkdir(exist_ok=True)

        existing_metrics = []
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                existing_metrics = json.load(f)

        existing_metrics.append(metrics)

        with open(metrics_file, "w") as f:
            json.dump(existing_metrics, f, indent=2)

    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED!")
        self.emergency_stop = True

        # Close all positions
        for symbol in list(self.positions.keys()):
            if self.positions[symbol]["quantity"] != 0:
                order = {
                    "symbol": symbol,
                    "side": "sell" if self.positions[symbol]["quantity"] > 0 else "buy",
                    "quantity": abs(self.positions[symbol]["quantity"]),
                    "type": "market",
                    "reason": "emergency_shutdown",
                }
                await self.order_manager.submit_order(order)

        # Cancel all pending orders
        await self.order_manager.cancel_all_orders()

        # Save final state
        self._save_shutdown_report()

        self.is_running = False
        logger.info("System shutdown complete")

    def _save_shutdown_report(self):
        """Save shutdown report"""
        {
            "timestamp": datetime.now().isoformat(),
            "reason": "emergency_shutdown",
            "final_pnl": self.daily_pnl,
            "final_capital": self.capital + self.daily_pnl,
            "positions_closed": len(self.positions),
            "orders_cancelled": len(self.orders),
        }

        report_file = Path("data/shutdown_reports.json")
        report_file.parent.mkdir(exist_ok=True)

        existing_reports = []
        if report_file.exists():
            with open(report_file, "r") as f:
                existing_reports = json.load(f)

        existing_reports.append(report)

        with open(report_file, "w") as f:
            json.dump(existing_reports, f, indent=2)

    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "emergency_stop": self.emergency_stop,
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "positions": len(self.positions),
            "symbols": len(self.symbols),
            "strategies": len(self.strategies),
        }


async def main():
    """Main entry point for live trading"""
    # Initialize system
    system = LiveTradingSystem()

    # Import strategy dependencies
    from src.strategies.traditional.momentum_strategy import MomentumStrategy
    from src.strategies.traditional.mean_reversion import MeanReversionStrategy
    from src.strategies.strategy_interface import StrategyConfig

    # Create strategy configurations
    momentum_config = StrategyConfig(
        name="MomentumStrategy",
        enabled=True,
        weight=0.5,
        risk_limit=0.02,
        max_positions=5,
        parameters={
            "rsi_period": 14,
            "rsi_buy_threshold": 60,
            "rsi_sell_threshold": 40,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "volume_period": 20,
            "volume_threshold": 1.5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "max_position_size": 0.1,
            "max_drawdown": 0.1,
            "position_sizing_method": "fixed",
        },
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    )

    mean_reversion_config = StrategyConfig(
        name="MeanReversionStrategy",
        enabled=True,
        weight=0.5,
        risk_limit=0.02,
        max_positions=5,
        parameters={
            "bb_period": 20,
            "bb_std": 2,
            "zscore_period": 20,
            "zscore_threshold": 2.0,
            "holding_period": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.03,
            "max_position_size": 0.1,
            "max_drawdown": 0.1,
            "position_sizing_method": "fixed",
        },
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    )

    # Initialize strategies with configs
    strategies = [MomentumStrategy(momentum_config), MeanReversionStrategy(mean_reversion_config)]

    # Define symbols to trade
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Start trading
    await system.start_trading(strategies, symbols, capital=10000)


if __name__ == "__main__":
    asyncio.run(main())
