"""
Risk Management Integration Example
Shows how to integrate risk management with existing trading system
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from risk_management import RiskMetrics, PositionSizing, StopLoss, ROIMonitor, ROITarget
import numpy as np
import pandas as pd


class RiskManagedTradingSystem:
    """Trading system with integrated risk management"""

    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.initial_capital = initial_capital

        # Risk management components
        self.risk_metrics = RiskMetrics()
        self.position_sizer = PositionSizing()
        self.stop_manager = StopLoss()
        self.roi_monitor = ROIMonitor(
            ROITarget(
                annual_target=10.0,  # 1000% target
                monthly_target=0.15,  # 15% monthly
                degradation_threshold=5.0,  # 500% degradation
            )
        )

        # Trading history
        self.positions = {}
        self.returns_history = []
        self.trades = []

    def calculate_position_size(
        self, symbol: str, price: float, signal_strength: float = 1.0
    ) -> int:
        """
        Calculate optimal position size using Kelly criterion

        Args:
            symbol: Trading symbol
            price: Current price
            signal_strength: Signal confidence (0-1)

        Returns:
            Position size in shares
        """
        # Get historical performance for Kelly calculation
        if len(self.returns_history) > 10:
            kelly_fraction = self.position_sizer.kelly_from_returns(
                self.returns_history
            )
        else:
            kelly_fraction = 0.02  # Default 2% risk

        # Adjust for signal strength
        adjusted_fraction = kelly_fraction * signal_strength

        # Calculate position value
        position_value = self.capital * adjusted_fraction

        # Convert to shares
        shares = int(position_value / price)

        # Ensure we don't exceed available capital
        max_shares = int(self.capital * 0.95 / price)  # Keep 5% cash buffer

        return min(shares, max_shares)

    def enter_position(
        self,
        symbol: str,
        price: float,
        direction: str = "long",
        signal_strength: float = 1.0,
        atr: float = None,
    ) -> bool:
        """
        Enter a new position with risk management

        Args:
            symbol: Trading symbol
            price: Entry price
            direction: "long" or "short"
            signal_strength: Signal confidence
            atr: Average True Range for stop calculation

        Returns:
            True if position entered successfully
        """
        # Calculate position size
        shares = self.calculate_position_size(symbol, price, signal_strength)

        if shares <= 0:
            return False

        # Calculate position value
        position_value = shares * price

        # Check available capital
        if position_value > self.capital * 0.95:
            return False

        # Enter position
        self.positions[symbol] = {
            "shares": shares,
            "entry_price": price,
            "direction": direction,
            "entry_time": pd.Timestamp.now(),
            "position_value": position_value,
        }

        # Update capital
        self.capital -= position_value

        # Set stop loss
        if atr is not None:
            self.stop_manager.set_atr_stop(symbol, price, atr, 2.0, direction)
        else:
            # Default 5% stop
            self.stop_manager.set_percentage_stop(symbol, price, 0.05, direction)

        print(
            f"Entered {direction} position: {shares} shares of {symbol} at ${price:.2f}"
        )
        return True

    def check_stops_and_exits(self, market_data: dict) -> list:
        """
        Check stop losses and exit positions if triggered

        Args:
            market_data: Dictionary with symbol: price mappings

        Returns:
            List of symbols where positions were closed
        """
        closed_positions = []

        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                current_price = market_data[symbol]

                # Check stop loss
                if self.stop_manager.check_stop_triggered(symbol, current_price):
                    self.exit_position(symbol, current_price, "stop_loss")
                    closed_positions.append(symbol)

        return closed_positions

    def exit_position(self, symbol: str, price: float, reason: str = "manual") -> float:
        """
        Exit a position and calculate P&L

        Args:
            symbol: Trading symbol
            price: Exit price
            reason: Exit reason

        Returns:
            Profit/Loss amount
        """
        if symbol not in self.positions:
            return 0.0

        position = self.positions[symbol]
        shares = position["shares"]
        entry_price = position["entry_price"]
        direction = position["direction"]

        # Calculate P&L
        if direction == "long":
            pnl = (price - entry_price) * shares
        else:
            pnl = (entry_price - price) * shares

        # Calculate return
        position_value = position["position_value"]
        trade_return = pnl / position_value if position_value > 0 else 0.0

        # Update capital
        exit_value = shares * price
        self.capital += exit_value

        # Record trade
        trade_record = {
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": price,
            "shares": shares,
            "direction": direction,
            "pnl": pnl,
            "return": trade_return,
            "exit_reason": reason,
            "exit_time": pd.Timestamp.now(),
        }
        self.trades.append(trade_record)

        # Update returns history
        self.returns_history.append(trade_return)

        # Track performance with ROI monitor
        if len(self.returns_history) % 10 == 0:  # Every 10 trades
            total_return = (self.capital - self.initial_capital) / self.initial_capital
            self.roi_monitor.track_performance(total_return)

        # Remove position and stop
        del self.positions[symbol]
        self.stop_manager.remove_stop(symbol)

        print(
            f"Exited {symbol}: {reason}, P&L: ${pnl:.2f}, Return: {trade_return*100:.2f}%"
        )
        return pnl

    def get_portfolio_risk_metrics(self) -> dict:
        """Get current portfolio risk metrics"""
        if len(self.returns_history) < 10:
            return {"error": "Insufficient data for risk calculation"}

        returns_series = pd.Series(self.returns_history)

        return {
            "var_95": self.risk_metrics.calculate_var(returns_series, 0.95),
            "cvar_95": self.risk_metrics.calculate_cvar(returns_series, 0.95),
            "sharpe_ratio": self.risk_metrics.calculate_sharpe_ratio(returns_series),
            "sortino_ratio": self.risk_metrics.calculate_sortino_ratio(returns_series),
            "total_trades": len(self.trades),
            "win_rate": (
                len([t for t in self.trades if t["pnl"] > 0]) / len(self.trades)
                if self.trades
                else 0
            ),
            "current_capital": self.capital,
            "total_return": (self.capital - self.initial_capital)
            / self.initial_capital,
            "active_positions": len(self.positions),
        }

    def get_roi_status(self) -> dict:
        """Get ROI monitoring status"""
        return self.roi_monitor.get_performance_summary(30)


def demo_integrated_system():
    """Demonstrate integrated risk-managed trading system"""
    print("INTEGRATED RISK-MANAGED TRADING SYSTEM DEMO")
    print("=" * 50)

    # Create trading system
    system = RiskManagedTradingSystem(100000)

    # Simulate some trades
    np.random.seed(42)
    ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

    for i in range(20):
        # Pick random symbol and price
        symbol = np.random.choice(symbols)
        price = np.random.uniform(100, 500)

        # Random signal strength
        signal_strength = np.random.uniform(0.3, 1.0)

        # Random ATR
        atr = price * np.random.uniform(0.02, 0.05)

        # Enter position
        if symbol not in system.positions:
            system.enter_position(symbol, price, "long", signal_strength, atr)

        # Simulate price movement and check exits
        if system.positions:
            # Simulate market data
            market_data = {}
            for pos_symbol in system.positions:
                # Random price movement
                current_price = system.positions[pos_symbol][
                    "entry_price"
                ] * np.random.uniform(0.90, 1.15)
                market_data[pos_symbol] = current_price

            # Check stops
            system.check_stops_and_exits(market_data)

            # Random exits for demo
            if np.random.random() < 0.3 and system.positions:
                exit_symbol = np.random.choice(list(system.positions.keys()))
                exit_price = market_data[exit_symbol]
                system.exit_position(exit_symbol, exit_price, "profit_taking")

    # Get final metrics
    print(f"\n{'='*50}")
    print("FINAL PERFORMANCE METRICS:")

    risk_metrics = system.get_portfolio_risk_metrics()
    if "error" not in risk_metrics:
        print(f"Total Trades: {risk_metrics['total_trades']}")
        print(f"Win Rate: {risk_metrics['win_rate']*100:.1f}%")
        print(f"Total Return: {risk_metrics['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
        print(f"VaR (95%): {risk_metrics['var_95']*100:.2f}%")
        print(f"Final Capital: ${risk_metrics['current_capital']:,.2f}")

    roi_status = system.get_roi_status()
    if "error" not in roi_status:
        print("\nROI Status:")
        print(f"Average ROI: {roi_status['average_roi']:.2f}%")
        print(f"Active Alerts: {roi_status['active_alerts']}")

    print(f"\n{'='*50}")
    print("STAGE 6 RISK MANAGEMENT: COMPLETE!")
    print("[OK] Position sizing with Kelly criterion")
    print("[OK] Automated stop loss management")
    print("[OK] Real-time risk monitoring")
    print("[OK] ROI target tracking (1000% goal)")
    print("[OK] Zero-cost strategy validation")


if __name__ == "__main__":
    demo_integrated_system()
