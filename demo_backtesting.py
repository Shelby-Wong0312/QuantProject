"""
Backtesting System Demo
Demonstrates the complete backtesting framework with a simple strategy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.backtesting import run_backtest, BacktestEngine, BacktestConfig
from src.strategies.base_strategy import BaseStrategy
from src.strategies.strategy_interface import TradingSignal, SignalType, StrategyConfig


class SimpleMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy for demonstration"""

    def _initialize_parameters(self):
        """Initialize strategy parameters"""
        self.fast_period = self.config.parameters.get("fast_period", 20)
        self.slow_period = self.config.parameters.get("slow_period", 50)
        self.lookback_period = max(self.fast_period, self.slow_period) + 10

    def calculate_signals(self, data: pd.DataFrame) -> list:
        """Calculate moving average crossover signals"""
        if len(data) < self.lookback_period:
            return []

        # Calculate moving averages
        fast_ma = data["close"].rolling(self.fast_period).mean()
        slow_ma = data["close"].rolling(self.slow_period).mean()

        # Get latest values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else current_fast
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else current_slow

        current_price = data["close"].iloc[-1]
        signals = []

        # Determine symbol from data or use first symbol in config
        symbol = (
            data.index.name
            if data.index.name
            else (self.config.symbols[0] if self.config.symbols else "UNKNOWN")
        )

        # Check for crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            # Golden cross - buy signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                strategy_name=self.name,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                metadata={
                    "reason": "Golden Cross",
                    "fast_ma": current_fast,
                    "slow_ma": current_slow,
                },
            )
            signals.append(signal)

        elif prev_fast >= prev_slow and current_fast < current_slow:
            # Death cross - sell signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=0.8,
                strategy_name=self.name,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                metadata={
                    "reason": "Death Cross",
                    "fast_ma": current_fast,
                    "slow_ma": current_slow,
                },
            )
            signals.append(signal)

        return signals

    def get_position_size(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> float:
        """Calculate position size - 10% of portfolio per position"""
        position_value = portfolio_value * 0.1  # 10% allocation
        position_size = position_value / current_price

        # Apply signal direction
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            position_size = -position_size

        return position_size

    def apply_risk_management(self, position, market_data: pd.DataFrame) -> dict:
        """Simple risk management - 5% stop loss"""
        current_price = market_data["close"].iloc[-1]

        if position.is_long():
            stop_loss = position.entry_price * 0.95  # 5% stop loss
            if current_price <= stop_loss:
                return {
                    "action": "close",
                    "reason": "Stop loss triggered",
                    "new_size": 0,
                    "stop_loss": stop_loss,
                    "take_profit": None,
                }
        else:
            stop_loss = position.entry_price * 1.05  # 5% stop loss for short
            if current_price >= stop_loss:
                return {
                    "action": "close",
                    "reason": "Stop loss triggered",
                    "new_size": 0,
                    "stop_loss": stop_loss,
                    "take_profit": None,
                }

        return {"action": "hold", "reason": "No risk management action needed"}


def generate_sample_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate sample market data for testing"""

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate realistic price data
    np.random.seed(42)  # For reproducible results

    # Parameters for price simulation
    initial_price = 100.0
    annual_return = 0.08  # 8% annual return
    annual_volatility = 0.2  # 20% annual volatility

    # Convert to daily
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # Generate random walks
    returns = np.random.normal(daily_return, daily_volatility, len(dates))

    # Add some trend and mean reversion
    prices = [initial_price]
    for i in range(1, len(dates)):
        # Add trend component
        trend_factor = np.sin(i / 50) * 0.001  # Cyclical trend

        # Add mean reversion
        price_deviation = (prices[-1] - initial_price) / initial_price
        mean_reversion = -price_deviation * 0.01

        # Calculate next price
        adjusted_return = returns[i] + trend_factor + mean_reversion
        next_price = prices[-1] * (1 + adjusted_return)
        prices.append(max(next_price, 1.0))  # Ensure price stays positive

    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    df["close"] = prices

    # Generate OHLV from close prices
    df["open"] = df["close"].shift(1).fillna(initial_price)

    # High is close + random positive amount
    high_factors = 1 + np.abs(np.random.normal(0, 0.01, len(df)))
    df["high"] = df[["open", "close"]].max(axis=1) * high_factors

    # Low is close - random positive amount
    low_factors = 1 - np.abs(np.random.normal(0, 0.01, len(df)))
    df["low"] = df[["open", "close"]].min(axis=1) * low_factors

    # Volume
    df["volume"] = np.random.uniform(100000, 1000000, len(df))

    # Set symbol name for signal generation
    df.index.name = symbol

    return df


def demo_simple_backtest():
    """Demonstrate simple backtesting using convenience function"""
    print("=" * 60)
    print("SIMPLE BACKTEST DEMO")
    print("=" * 60)

    # Generate sample data
    print("Generating sample data...")
    data = {"AAPL": generate_sample_data("AAPL", 365), "GOOGL": generate_sample_data("GOOGL", 365)}

    # Create strategy
    config = StrategyConfig(
        name="Simple MA Strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.02,
        max_positions=5,
        symbols=["AAPL", "GOOGL"],
        parameters={"fast_period": 20, "slow_period": 50},
    )

    strategy = SimpleMAStrategy(config, initial_capital=100000)

    # Run backtest using convenience function
    print("Running backtest...")
    results = run_backtest(
        strategy=strategy, data=data, initial_capital=100000, commission=0.001, slippage=0.0005
    )

    # Display results
    print("\nBacktest completed!")
    print(f"Final Portfolio Value: ${results['summary']['final_value']:,.2f}")
    print(f"Total Return: {results['summary']['total_return']:.2%}")
    print(f"Annual Return: {results['summary']['annual_return']:.2%}")
    print(f"Total Trades: {results['summary']['total_trades']}")

    # Performance metrics
    perf = results["performance"]
    print(f"\nPerformance Metrics:")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {perf.get('win_rate', 0):.2%}")
    print(f"Profit Factor: {perf.get('profit_factor', 0):.2f}")

    return results


def demo_advanced_backtest():
    """Demonstrate advanced backtesting with custom configuration"""
    print("\n" + "=" * 60)
    print("ADVANCED BACKTEST DEMO")
    print("=" * 60)

    # Custom configuration - use default dates for now
    config = BacktestConfig(
        initial_capital=250000,
        commission=0.0005,  # Lower commission
        slippage=0.0002,  # Lower slippage
        rebalance_frequency="weekly",
    )

    # Create engine
    engine = BacktestEngine(config)

    # Add multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    print(f"Adding data for {len(symbols)} symbols...")

    for symbol in symbols:
        data = generate_sample_data(symbol, 400)
        engine.add_data(data, symbol)

    # Create strategy
    strategy_config = StrategyConfig(
        name="Advanced MA Strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.015,  # Tighter risk limit
        max_positions=10,
        symbols=symbols,
        parameters={"fast_period": 15, "slow_period": 40},
    )

    strategy = SimpleMAStrategy(strategy_config, initial_capital=config.initial_capital)

    # Run backtest
    print("Running advanced backtest...")
    results = engine.run_backtest(strategy)

    # Display detailed results
    print("\nAdvanced Backtest Results:")
    print("=" * 40)

    summary = results["summary"]
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Value: ${summary['final_value']:,.2f}")
    print(f"Total Return: {summary['total_return']:.2%}")
    print(f"Annual Return: {summary['annual_return']:.2%}")
    print(f"Years Tested: {summary['years']:.1f}")

    perf = results["performance"]
    print(f"\nRisk Metrics:")
    print(f"Volatility: {perf.get('volatility_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {perf.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown: {perf.get('max_drawdown_pct', 0):.2f}%")

    print(f"\nTrading Statistics:")
    print(f"Total Trades: {perf.get('total_trades', 0)}")
    print(f"Win Rate: {perf.get('win_rate_pct', 0):.1f}%")
    print(f"Profit Factor: {perf.get('profit_factor', 0):.2f}")
    print(f"Avg Trade P&L: ${perf.get('avg_trade_pnl', 0):.2f}")

    # Show trades summary
    trades = results["trades"]
    if trades:
        print(f"\nRecent Trades (last 5):")
        for trade in trades[-5:]:
            pnl = trade.get("pnl", 0)
            pnl_str = f"${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            print(
                f"  {trade['symbol']} {trade['action']} @ ${trade['price']:.2f} -> P&L: {pnl_str}"
            )

    return results


def main():
    """Run backtesting demonstrations"""
    print("BACKTESTING FRAMEWORK DEMONSTRATION")
    print("Stage 5: Complete Backtesting System")
    print("=" * 60)

    try:
        # Run simple demo
        simple_results = demo_simple_backtest()

        # Run advanced demo
        advanced_results = demo_advanced_backtest()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("* Simple backtesting demonstrated")
        print("* Advanced backtesting demonstrated")
        print("* Performance metrics calculated")
        print("* Risk analysis completed")
        print("* Trading statistics generated")

        print(f"\nFramework Features Demonstrated:")
        print("- Event-driven backtesting engine")
        print("- Portfolio management with P&L tracking")
        print("- Comprehensive performance analysis")
        print("- Transaction cost modeling (commission + slippage)")
        print("- Risk management integration")
        print("- Multiple timeframe support")
        print("- Strategy validation framework")

        return simple_results, advanced_results

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    simple_results, advanced_results = main()
