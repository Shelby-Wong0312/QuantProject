"""
Comprehensive Backtest for ALL Stocks in Database
Test top 5 indicators on every available stock
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import json
import time
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# Import top 5 performing indicators based on previous results
from src.indicators.momentum_indicators import CCI, WilliamsR, Stochastic
from src.indicators.volume_indicators import VolumeSMA, OBV


class ComprehensiveBacktest:
    """Run backtest on ALL available stocks"""

    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quant_trading.db"
        )
        self.initial_capital = 100000.0
        self.commission_rate = 0.001
        self.slippage_rate = 0.0005

        # Top 5 indicators from previous test
        self.indicators = [
            ("CCI_20", CCI(period=20)),
            ("Williams_R", WilliamsR(period=14)),
            ("Stochastic", Stochastic(k_period=14, d_period=3)),
            ("VolumeSMA", VolumeSMA(period=20)),
            ("OBV", OBV()),
        ]

    def get_all_stocks(self):
        """Get all stocks with sufficient data"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT symbol, COUNT(*) as data_points
            FROM daily_data
            GROUP BY symbol
            HAVING COUNT(*) >= 200
            ORDER BY symbol
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df["symbol"].tolist(), df.set_index("symbol")["data_points"].to_dict()

    def backtest_stock(self, symbol: str, indicator_name: str, indicator) -> Dict:
        """Backtest single stock with single indicator"""

        conn = sqlite3.connect(self.db_path)

        # Get stock data
        query = f"""
            SELECT date, open_price as open, high_price as high, 
                   low_price as low, close_price as close, volume
            FROM daily_data
            WHERE symbol = '{symbol}'
            ORDER BY date DESC
            LIMIT 504
        """

        try:
            df = pd.read_sql_query(query, conn, parse_dates=["date"])
            conn.close()

            if len(df) < 200:
                return None

            df.set_index("date", inplace=True)
            df = df.sort_index()

            # Generate signals
            signals = indicator.get_signals(df)

            # Run backtest
            portfolio = {
                "cash": self.initial_capital,
                "shares": 0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            }

            values = []
            buy_price = 0

            for i in range(len(df)):
                price = df["close"].iloc[i]

                if i >= len(signals):
                    break

                # Check signals
                if "buy" in signals.columns and signals["buy"].iloc[i] and portfolio["cash"] > 0:
                    # Buy signal
                    shares_to_buy = int(
                        portfolio["cash"]
                        / (price * (1 + self.commission_rate + self.slippage_rate))
                    )
                    if shares_to_buy > 0:
                        cost = (
                            shares_to_buy * price * (1 + self.commission_rate + self.slippage_rate)
                        )
                        portfolio["cash"] -= cost
                        portfolio["shares"] += shares_to_buy
                        portfolio["trades"] += 1
                        buy_price = price

                elif (
                    "sell" in signals.columns
                    and signals["sell"].iloc[i]
                    and portfolio["shares"] > 0
                ):
                    # Sell signal
                    revenue = (
                        portfolio["shares"]
                        * price
                        * (1 - self.commission_rate - self.slippage_rate)
                    )
                    portfolio["cash"] += revenue

                    # Track wins/losses
                    if price > buy_price:
                        portfolio["wins"] += 1
                    else:
                        portfolio["losses"] += 1

                    portfolio["shares"] = 0

                # Calculate portfolio value
                total_value = portfolio["cash"] + portfolio["shares"] * price
                values.append(total_value)

            if len(values) == 0:
                return None

            # Calculate metrics
            values_series = pd.Series(values, index=df.index[: len(values)])
            returns = values_series.pct_change().dropna()

            total_return = (values[-1] / self.initial_capital - 1) * 100

            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe_ratio = 0

            # Max drawdown
            cummax = values_series.cummax()
            drawdown = (values_series - cummax) / cummax
            max_drawdown = abs(drawdown.min()) * 100

            # Win rate
            total_trades = portfolio["wins"] + portfolio["losses"]
            win_rate = (portfolio["wins"] / total_trades * 100) if total_trades > 0 else 0

            return {
                "symbol": symbol,
                "indicator": indicator_name,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "num_trades": portfolio["trades"],
                "win_rate": win_rate,
                "final_value": values[-1],
            }

        except Exception as e:
            conn.close()
            return None

    def run_comprehensive_backtest(self):
        """Run backtest on ALL stocks"""

        print("=" * 80)
        print("COMPREHENSIVE BACKTEST - ALL STOCKS")
        print("=" * 80)
        print(f"Start Time: {datetime.now()}")
        print("-" * 80)

        # Get all stocks
        all_stocks, data_points = self.get_all_stocks()
        print(f"Found {len(all_stocks)} stocks with sufficient data")
        print(f"Testing {len(self.indicators)} indicators on each stock")
        print(f"Total backtests to run: {len(all_stocks) * len(self.indicators)}")
        print("-" * 80)

        # Store all results
        all_results = []
        stock_performance = {}
        indicator_performance = {ind[0]: [] for ind in self.indicators}

        # Progress tracking
        total_tests = len(all_stocks) * len(self.indicators)
        completed = 0
        start_time = time.time()

        # Process in batches for better progress reporting
        batch_size = 10

        for batch_start in range(0, len(all_stocks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_stocks))
            batch_stocks = all_stocks[batch_start:batch_end]

            print(
                f"\nProcessing batch {batch_start//batch_size + 1}/{(len(all_stocks)-1)//batch_size + 1} "
                f"(Stocks {batch_start+1}-{batch_end}/{len(all_stocks)})"
            )

            for symbol in batch_stocks:
                stock_results = []

                for indicator_name, indicator in self.indicators:
                    result = self.backtest_stock(symbol, indicator_name, indicator)

                    if result:
                        all_results.append(result)
                        stock_results.append(result)
                        indicator_performance[indicator_name].append(result["total_return"])

                    completed += 1

                    # Progress update every 50 tests
                    if completed % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = (total_tests - completed) / rate
                        print(
                            f"  Progress: {completed}/{total_tests} ({completed/total_tests*100:.1f}%) "
                            f"- Est. remaining: {remaining/60:.1f} min"
                        )

                # Store best result for this stock
                if stock_results:
                    best_result = max(stock_results, key=lambda x: x["total_return"])
                    stock_performance[symbol] = best_result

        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE - ANALYZING RESULTS")
        print("=" * 80)

        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_results)

        if len(results_df) == 0:
            print("No valid results found!")
            return

        # Save raw results
        results_df.to_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "reports",
                "all_stocks_backtest_raw.csv",
            ),
            index=False,
        )

        # Analyze by indicator
        print("\n1. INDICATOR PERFORMANCE SUMMARY")
        print("-" * 80)
        print(
            f"{'Indicator':<15} {'Avg Return':<12} {'Best Return':<12} {'Worst Return':<12} "
            f"{'Positive %':<12} {'Avg Trades':<12}"
        )
        print("-" * 80)

        indicator_summary = {}
        for ind_name, _ in self.indicators:
            ind_results = results_df[results_df["indicator"] == ind_name]
            if len(ind_results) > 0:
                avg_return = ind_results["total_return"].mean()
                best_return = ind_results["total_return"].max()
                worst_return = ind_results["total_return"].min()
                positive_pct = (ind_results["total_return"] > 0).mean() * 100
                avg_trades = ind_results["num_trades"].mean()

                print(
                    f"{ind_name:<15} {avg_return:>11.2f}% {best_return:>11.2f}% "
                    f"{worst_return:>11.2f}% {positive_pct:>11.2f}% {avg_trades:>11.1f}"
                )

                indicator_summary[ind_name] = {
                    "avg_return": avg_return,
                    "best_return": best_return,
                    "worst_return": worst_return,
                    "positive_pct": positive_pct,
                    "avg_trades": avg_trades,
                    "total_stocks": len(ind_results),
                }

        # Find top performing stocks
        print("\n2. TOP 20 PERFORMING STOCKS")
        print("-" * 80)

        top_stocks = results_df.nlargest(20, "total_return")
        print(
            f"{'Rank':<5} {'Symbol':<10} {'Indicator':<15} {'Return':<12} {'Sharpe':<10} "
            f"{'Max DD':<10} {'Win Rate':<10}"
        )
        print("-" * 80)

        for i, row in enumerate(top_stocks.itertuples(), 1):
            print(
                f"{i:<5} {row.symbol:<10} {row.indicator:<15} {row.total_return:>10.2f}% "
                f"{row.sharpe_ratio:>9.2f} {row.max_drawdown:>9.2f}% {row.win_rate:>9.2f}%"
            )

        # Find stocks that work well with multiple indicators
        print("\n3. STOCKS PERFORMING WELL ACROSS MULTIPLE INDICATORS")
        print("-" * 80)

        # Group by symbol and calculate average performance
        symbol_avg = (
            results_df.groupby("symbol")
            .agg({"total_return": "mean", "sharpe_ratio": "mean", "win_rate": "mean"})
            .round(2)
        )

        # Count positive returns per stock
        positive_counts = results_df[results_df["total_return"] > 0].groupby("symbol").size()
        symbol_avg["positive_indicators"] = positive_counts

        # Filter stocks that work with at least 3 indicators
        multi_indicator_stocks = (
            symbol_avg[symbol_avg["positive_indicators"] >= 3]
            .sort_values("total_return", ascending=False)
            .head(20)
        )

        print(
            f"{'Symbol':<10} {'Avg Return':<12} {'Avg Sharpe':<12} {'Avg Win Rate':<12} {'Positive Ind':<12}"
        )
        print("-" * 80)

        for symbol, row in multi_indicator_stocks.iterrows():
            print(
                f"{symbol:<10} {row['total_return']:>11.2f}% {row['sharpe_ratio']:>11.2f} "
                f"{row['win_rate']:>11.2f}% {int(row['positive_indicators']):>11}"
            )

        # Statistical summary
        print("\n4. OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total stocks tested: {len(all_stocks)}")
        print(f"Total backtests run: {len(results_df)}")
        print(f"Average return across all tests: {results_df['total_return'].mean():.2f}%")
        print(
            f"Percentage of profitable tests: {(results_df['total_return'] > 0).mean() * 100:.2f}%"
        )
        print(
            f"Best single result: {results_df['total_return'].max():.2f}% "
            f"({results_df.loc[results_df['total_return'].idxmax(), 'symbol']} with "
            f"{results_df.loc[results_df['total_return'].idxmax(), 'indicator']})"
        )

        # Save comprehensive report
        report = {
            "test_date": datetime.now().isoformat(),
            "total_stocks": len(all_stocks),
            "total_tests": len(results_df),
            "indicator_summary": indicator_summary,
            "top_20_results": top_stocks.to_dict("records"),
            "multi_indicator_stocks": multi_indicator_stocks.to_dict("index"),
            "overall_stats": {
                "avg_return": results_df["total_return"].mean(),
                "median_return": results_df["total_return"].median(),
                "std_return": results_df["total_return"].std(),
                "profitable_pct": (results_df["total_return"] > 0).mean() * 100,
                "best_return": results_df["total_return"].max(),
                "worst_return": results_df["total_return"].min(),
            },
        }

        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports",
            "comprehensive_backtest_report.json",
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")
        print(f"Raw data saved to: all_stocks_backtest_raw.csv")

        # Create list of recommended stocks
        print("\n" + "=" * 80)
        print("RECOMMENDED PORTFOLIO")
        print("=" * 80)

        # Get top 10 stocks by average return with multiple indicators
        recommended = multi_indicator_stocks.head(10)

        print("\nTop 10 Recommended Stocks (work well with multiple indicators):")
        for i, (symbol, row) in enumerate(recommended.iterrows(), 1):
            print(
                f"{i:2}. {symbol:<8} - Avg Return: {row['total_return']:>6.2f}% | "
                f"Works with {int(row['positive_indicators'])} indicators"
            )

        elapsed_time = time.time() - start_time
        print(f"\nTotal processing time: {elapsed_time/60:.1f} minutes")
        print(f"Average time per backtest: {elapsed_time/len(results_df):.2f} seconds")

        print("\n" + "=" * 80)
        print("COMPREHENSIVE BACKTEST COMPLETE!")
        print("=" * 80)

        return results_df, report


if __name__ == "__main__":
    backtester = ComprehensiveBacktest()
    results_df, report = backtester.run_comprehensive_backtest()
