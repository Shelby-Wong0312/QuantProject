"""
Comprehensive Backtesting for All Technical Indicators
Test each indicator's profitability and generate comparison report
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import json
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# Import all indicators
from src.indicators.trend_indicators import SMA, EMA, WMA, VWAP
from src.indicators.momentum_indicators import RSI, MACD, Stochastic, WilliamsR, CCI
from src.indicators.volatility_indicators import (
    BollingerBands,
    ATR,
    KeltnerChannel,
    DonchianChannel,
)
from src.indicators.volume_indicators import OBV, VolumeSMA, MFI, ADLine


class IndicatorBacktest:
    """Backtest engine for single indicator strategies"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005  # 0.05% slippage

    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """Run backtest on given signals"""

        # Initialize portfolio
        portfolio = {
            "cash": self.initial_capital,
            "shares": 0,
            "total_value": self.initial_capital,
            "trades": [],
            "daily_returns": [],
        }

        values = []
        positions = []

        for i in range(len(data)):
            date = data.index[i]
            price = data["close"].iloc[i]

            # Skip if no signal
            if i >= len(signals):
                continue

            # Get signal
            if "buy" in signals.columns:
                buy_signal = signals["buy"].iloc[i] if i < len(signals) else False
                sell_signal = signals["sell"].iloc[i] if i < len(signals) else False
            else:
                buy_signal = False
                sell_signal = False

            # Execute trades
            if buy_signal and portfolio["cash"] > 0:
                # Buy signal - invest all cash
                shares_to_buy = int(
                    portfolio["cash"]
                    / (price * (1 + self.commission_rate + self.slippage_rate))
                )
                if shares_to_buy > 0:
                    cost = (
                        shares_to_buy
                        * price
                        * (1 + self.commission_rate + self.slippage_rate)
                    )
                    portfolio["cash"] -= cost
                    portfolio["shares"] += shares_to_buy
                    portfolio["trades"].append(
                        {
                            "date": date,
                            "type": "BUY",
                            "price": price,
                            "shares": shares_to_buy,
                            "cost": cost,
                        }
                    )

            elif sell_signal and portfolio["shares"] > 0:
                # Sell signal - sell all shares
                revenue = (
                    portfolio["shares"]
                    * price
                    * (1 - self.commission_rate - self.slippage_rate)
                )
                portfolio["cash"] += revenue
                portfolio["trades"].append(
                    {
                        "date": date,
                        "type": "SELL",
                        "price": price,
                        "shares": portfolio["shares"],
                        "revenue": revenue,
                    }
                )
                portfolio["shares"] = 0

            # Calculate portfolio value
            portfolio["total_value"] = portfolio["cash"] + portfolio["shares"] * price
            values.append(portfolio["total_value"])
            positions.append(1 if portfolio["shares"] > 0 else 0)

        # Calculate performance metrics
        values = pd.Series(values, index=data.index[: len(values)])
        returns = values.pct_change().dropna()

        # Calculate metrics
        total_return = (values.iloc[-1] / self.initial_capital - 1) * 100
        num_trades = len(portfolio["trades"])

        if len(returns) > 0:
            sharpe_ratio = (
                np.sqrt(252) * returns.mean() / returns.std()
                if returns.std() > 0
                else 0
            )
            max_drawdown = self.calculate_max_drawdown(values)
            win_rate = self.calculate_win_rate(portfolio["trades"])
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0

        # Annualized return
        years = len(data) / 252
        annualized_return = (
            ((values.iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100
            if years > 0
            else 0
        )

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "final_value": values.iloc[-1],
            "values": values,
            "positions": positions,
        }

    def calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        return abs(drawdown.min()) * 100

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if len(trades) < 2:
            return 0

        profits = []
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                if buy_trade["type"] == "BUY" and sell_trade["type"] == "SELL":
                    profit = sell_trade["revenue"] - buy_trade["cost"]
                    profits.append(profit)

        if len(profits) == 0:
            return 0

        wins = sum(1 for p in profits if p > 0)
        return (wins / len(profits)) * 100


def test_all_indicators_backtest():
    """Test all indicators with backtesting"""

    print("=" * 80)
    print("COMPREHENSIVE INDICATOR BACKTESTING")
    print("=" * 80)
    print(f"Start Time: {datetime.now()}")
    print("-" * 80)

    # Get database connection
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "quant_trading.db",
    )
    conn = sqlite3.connect(db_path)

    # Test on multiple stocks
    test_symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "AMZN",
        "NVDA",
        "TSLA",
        "JPM",
        "V",
        "WMT",
    ]

    # Initialize results storage
    all_results = {}

    # Initialize backtester
    backtester = IndicatorBacktest(initial_capital=100000)

    # Define indicators to test
    indicators_to_test = [
        # Trend Indicators
        ("SMA_20", SMA(period=20)),
        ("SMA_50", SMA(period=50)),
        ("EMA_20", EMA(period=20)),
        ("EMA_50", EMA(period=50)),
        ("WMA_20", WMA(period=20)),
        ("VWAP", VWAP()),
        # Momentum Indicators
        ("RSI_14", RSI(period=14)),
        ("MACD", MACD()),
        ("Stochastic", Stochastic()),
        ("Williams_R", WilliamsR()),
        ("CCI_20", CCI(period=20)),
        # Volatility Indicators
        ("BollingerBands", BollingerBands()),
        ("ATR_14", ATR(period=14)),
        ("KeltnerChannel", KeltnerChannel()),
        ("DonchianChannel", DonchianChannel()),
        # Volume Indicators
        ("OBV", OBV()),
        ("VolumeSMA", VolumeSMA()),
        ("MFI_14", MFI(period=14)),
        ("ADLine", ADLine()),
    ]

    print(
        f"Testing {len(indicators_to_test)} indicators on {len(test_symbols)} stocks..."
    )
    print("-" * 80)

    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")

        # Get stock data (2 years for faster testing)
        query = """
            SELECT date, open_price as open, high_price as high, 
                   low_price as low, close_price as close, volume
            FROM daily_data
            WHERE symbol = '{symbol}'
            ORDER BY date DESC
            LIMIT 504
        """

        try:
            df = pd.read_sql_query(query, conn, parse_dates=["date"])
            if len(df) < 200:
                print(f"  Skipping {symbol} - insufficient data ({len(df)} days)")
                continue

            df.set_index("date", inplace=True)
            df = df.sort_index()

            symbol_results = {}

            for indicator_name, indicator in indicators_to_test:
                try:
                    # Generate signals
                    indicator.get_signals(df)

                    # Run backtest
                    results = backtester.run_backtest(df, signals)

                    symbol_results[indicator_name] = {
                        "total_return": results["total_return"],
                        "annualized_return": results["annualized_return"],
                        "sharpe_ratio": results["sharpe_ratio"],
                        "max_drawdown": results["max_drawdown"],
                        "num_trades": results["num_trades"],
                        "win_rate": results["win_rate"],
                    }

                except Exception as e:
                    print(f"  Error testing {indicator_name}: {str(e)}")
                    symbol_results[indicator_name] = {
                        "total_return": 0,
                        "annualized_return": 0,
                        "sharpe_ratio": 0,
                        "max_drawdown": 0,
                        "num_trades": 0,
                        "win_rate": 0,
                    }

            all_results[symbol] = symbol_results

            # Print best performer for this stock
            best_indicator = max(
                symbol_results.items(), key=lambda x: x[1]["total_return"]
            )
            print(
                f"  Best: {best_indicator[0]} - Return: {best_indicator[1]['total_return']:.2f}%"
            )

        except Exception as e:
            print(f"  Error processing {symbol}: {str(e)}")

    conn.close()

    # Aggregate results across all stocks
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS - AVERAGE PERFORMANCE ACROSS ALL STOCKS")
    print("=" * 80)

    aggregate_results = {}

    for indicator_name, _ in indicators_to_test:
        returns = []
        sharpes = []
        drawdowns = []
        win_rates = []
        trades = []

        for symbol in all_results:
            if indicator_name in all_results[symbol]:
                returns.append(all_results[symbol][indicator_name]["total_return"])
                sharpes.append(all_results[symbol][indicator_name]["sharpe_ratio"])
                drawdowns.append(all_results[symbol][indicator_name]["max_drawdown"])
                win_rates.append(all_results[symbol][indicator_name]["win_rate"])
                trades.append(all_results[symbol][indicator_name]["num_trades"])

        if returns:
            aggregate_results[indicator_name] = {
                "avg_return": np.mean(returns),
                "avg_sharpe": np.mean(sharpes),
                "avg_drawdown": np.mean(drawdowns),
                "avg_win_rate": np.mean(win_rates),
                "avg_trades": np.mean(trades),
                "tested_stocks": len(returns),
            }

    # Sort by average return
    sorted_results = sorted(
        aggregate_results.items(), key=lambda x: x[1]["avg_return"], reverse=True
    )

    # Print ranking table
    print("\nRANKING BY AVERAGE RETURN:")
    print("-" * 80)
    print(
        f"{'Rank':<5} {'Indicator':<20} {'Avg Return':<12} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10} {'Trades':<8}"
    )
    print("-" * 80)

    for i, (indicator, metrics) in enumerate(sorted_results, 1):
        print(
            f"{i:<5} {indicator:<20} {metrics['avg_return']:>10.2f}% {metrics['avg_sharpe']:>9.2f} "
            f"{metrics['avg_drawdown']:>9.2f}% {metrics['avg_win_rate']:>9.2f}% {metrics['avg_trades']:>7.1f}"
        )

    # Save detailed results to JSON
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports",
        "indicator_backtest_results.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    report_data = {
        "test_date": datetime.now().isoformat(),
        "test_symbols": test_symbols,
        "aggregate_results": aggregate_results,
        "detailed_results": all_results,
        "best_indicator": sorted_results[0][0] if sorted_results else None,
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 80)
    print("WINNER ANNOUNCEMENT")
    print("=" * 80)

    if sorted_results:
        winner = sorted_results[0]
        print(f"\n*** MOST PROFITABLE INDICATOR: {winner[0]} ***")
        print(f"   Average Return: {winner[1]['avg_return']:.2f}%")
        print(f"   Sharpe Ratio: {winner[1]['avg_sharpe']:.2f}")
        print(f"   Max Drawdown: {winner[1]['avg_drawdown']:.2f}%")
        print(f"   Win Rate: {winner[1]['avg_win_rate']:.2f}%")
        print(f"   Avg Trades: {winner[1]['avg_trades']:.1f}")

    print(f"\nDetailed results saved to: {report_path}")
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE!")
    print("=" * 80)

    return aggregate_results


if __name__ == "__main__":
    results = test_all_indicators_backtest()
