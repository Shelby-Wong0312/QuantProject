"""
Quick Backtest on Sample Stocks with Full 15 Years Data
Shows immediate progress and results
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

from src.indicators.momentum_indicators import CCI

print("=" * 80)
print("QUICK BACKTEST - SAMPLE STOCKS WITH 15 YEARS DATA")
print("=" * 80)
print(f"Start Time: {datetime.now()}")

# Database connection
db_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quant_trading.db"
)
conn = sqlite3.connect(db_path)

# Get top 100 stocks by data availability
print("\nStep 1: Getting stock list...")
query = """
    SELECT symbol, COUNT(*) as data_points
    FROM daily_data
    GROUP BY symbol
    HAVING COUNT(*) >= 1000
    ORDER BY COUNT(*) DESC
    LIMIT 100
"""

stocks_df = pd.read_sql_query(query, conn)
print(f"Found {len(stocks_df)} stocks with 1000+ days of data")

# Initialize CCI indicator
cci = CCI(period=20)
results = []

print("\nStep 2: Running backtest...")
print("-" * 80)

for idx, row in enumerate(stocks_df.itertuples()):
    symbol = row.symbol

    # Get ALL historical data for this stock
    query = f"""
        SELECT date, close_price as close, high_price as high, 
               low_price as low, open_price as open, volume
        FROM daily_data
        WHERE symbol = '{symbol}'
        ORDER BY date ASC
    """

    try:
        df = pd.read_sql_query(query, conn, parse_dates=["date"])

        if len(df) < 100:
            continue

        df.set_index("date", inplace=True)

        # Calculate date range
        date_range = (df.index[-1] - df.index[0]).days / 365.25

        # Generate CCI signals
        signals = cci.get_signals(df)

        # Simple backtest
        initial_capital = 100000
        cash = initial_capital
        shares = 0
        trades = 0

        for i in range(len(df)):
            if i >= len(signals):
                break

            price = df["close"].iloc[i]

            if signals["buy"].iloc[i] and cash > 0:
                shares = cash / price
                cash = 0
                trades += 1
            elif signals["sell"].iloc[i] and shares > 0:
                cash = shares * price
                shares = 0

        # Final value
        final_value = cash + shares * df["close"].iloc[-1]
        total_return = (final_value / initial_capital - 1) * 100
        annual_return = (
            (((final_value / initial_capital) ** (1 / date_range)) - 1) * 100
            if date_range > 0
            else 0
        )

        results.append(
            {
                "symbol": symbol,
                "total_return": total_return,
                "annual_return": annual_return,
                "trades": trades,
                "years": date_range,
                "data_points": len(df),
                "final_value": final_value,
            }
        )

        # Print progress every 10 stocks
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(stocks_df)} stocks...")
            if results:
                avg_return = np.mean([r["total_return"] for r in results])
                print(f"  Current avg return: {avg_return:.2f}%")

    except Exception as e:
        continue

conn.close()

print("\n" + "=" * 80)
print("BACKTEST RESULTS - 15 YEARS DATA")
print("=" * 80)

# Convert to DataFrame
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("No valid results!")
else:
    # Sort by total return
    results_df = results_df.sort_values("total_return", ascending=False)

    # Save results
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports",
        "sample_15years_backtest.csv",
    )
    results_df.to_csv(csv_path, index=False)

    print(f"\n1. TOP 20 PERFORMERS (15 YEARS):")
    print("-" * 80)
    print(
        f"{'Rank':<5} {'Symbol':<8} {'Total Return':<15} {'Annual Return':<15} {'Years':<8} {'Trades':<8}"
    )
    print("-" * 80)

    for i, row in enumerate(results_df.head(20).itertuples(), 1):
        print(
            f"{i:<5} {row.symbol:<8} {row.total_return:>13.2f}% {row.annual_return:>13.2f}% "
            f"{row.years:>7.1f} {row.trades:>7}"
        )

    print("\n2. STATISTICS:")
    print("-" * 80)
    print(f"Stocks tested: {len(results_df)}")
    print(f"Average total return: {results_df['total_return'].mean():.2f}%")
    print(f"Average annual return: {results_df['annual_return'].mean():.2f}%")
    print(
        f"Best performer: {results_df.iloc[0]['symbol']} ({results_df.iloc[0]['total_return']:.2f}%)"
    )
    print(
        f"Profitable stocks: {len(results_df[results_df['total_return'] > 0])} "
        f"({len(results_df[results_df['total_return'] > 0])/len(results_df)*100:.1f}%)"
    )

    # Stocks with >100% return
    high_performers = results_df[results_df["total_return"] > 100]
    print(f"\nStocks with >100% return (15 years): {len(high_performers)}")

    if len(high_performers) > 0:
        print("\nHIGH PERFORMERS (>100% total return):")
        for row in high_performers.head(10).itertuples():
            print(f"  {row.symbol}: {row.total_return:.2f}% total, {row.annual_return:.2f}% annual")

    print(f"\nResults saved to: {csv_path}")

print("\n" + "=" * 80)
print("QUICK BACKTEST COMPLETE!")
print("=" * 80)
