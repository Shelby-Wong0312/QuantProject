"""
Generate Summary Report for Technical Indicators
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import sqlite3
from datetime import datetime
import json


def generate_indicator_summary():
    """Generate comprehensive indicator summary"""

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "quant_trading.db",
    )
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("TECHNICAL INDICATOR SUMMARY REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Check if indicators table exists
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='trend_indicators'
    """
    )

    if not cursor.fetchone():
        print("No indicator data found. Please run calculate_indicators.py first.")
        return

    # Count stocks with indicators
    query = """
        SELECT COUNT(DISTINCT symbol) as total_stocks,
               COUNT(*) as total_records,
               MIN(date) as start_date,
               MAX(date) as end_date
        FROM trend_indicators
    """
    stats = pd.read_sql_query(query, conn)

    print(f"Stocks with indicators: {stats['total_stocks'].iloc[0]}")
    print(f"Total indicator records: {stats['total_records'].iloc[0]:,}")
    print(f"Date range: {stats['start_date'].iloc[0]} to {stats['end_date'].iloc[0]}")
    print("-" * 60)

    # Get latest signals
    query = """
        SELECT date, COUNT(DISTINCT symbol) as golden_cross_count
        FROM trend_indicators
        WHERE golden_cross = 1
        GROUP BY date
        ORDER BY date DESC
        LIMIT 10
    """
    golden_crosses = pd.read_sql_query(query, conn)

    if not golden_crosses.empty:
        print("\nRecent Golden Cross Signals (Bullish):")
        for _, row in golden_crosses.iterrows():
            print(f"  {row['date']}: {row['golden_cross_count']} stocks")

    query = """
        SELECT date, COUNT(DISTINCT symbol) as death_cross_count
        FROM trend_indicators
        WHERE death_cross = 1
        GROUP BY date
        ORDER BY date DESC
        LIMIT 10
    """
    death_crosses = pd.read_sql_query(query, conn)

    if not death_crosses.empty:
        print("\nRecent Death Cross Signals (Bearish):")
        for _, row in death_crosses.iterrows():
            print(f"  {row['date']}: {row['death_cross_count']} stocks")

    # Get current trend distribution
    query = """
        SELECT trend_direction, COUNT(DISTINCT symbol) as count
        FROM trend_indicators
        WHERE date = (SELECT MAX(date) FROM trend_indicators)
        GROUP BY trend_direction
    """
    trend_dist = pd.read_sql_query(query, conn)

    if not trend_dist.empty:
        print("\nCurrent Market Trend Distribution:")
        for _, row in trend_dist.iterrows():
            print(f"  {row['trend_direction'].capitalize()}: {row['count']} stocks")

    # Get stocks with strongest trends (EMA50 furthest from EMA200)
    query = """
        SELECT symbol, 
               ema_50, 
               ema_200,
               ABS(ema_50 - ema_200) / ema_200 * 100 as trend_strength,
               trend_direction
        FROM trend_indicators
        WHERE date = (SELECT MAX(date) FROM trend_indicators)
        AND ema_50 IS NOT NULL 
        AND ema_200 IS NOT NULL
        ORDER BY trend_strength DESC
        LIMIT 10
    """
    strong_trends = pd.read_sql_query(query, conn)

    if not strong_trends.empty:
        print("\nStocks with Strongest Trends:")
        for _, row in strong_trends.iterrows():
            print(
                f"  {row['symbol']}: {row['trend_strength']:.1f}% ({row['trend_direction']})"
            )

    # Find stocks near support/resistance (price near major MAs)
    query = """
        SELECT t.symbol,
               d.close_price,
               t.sma_200,
               ABS(d.close_price - t.sma_200) / t.sma_200 * 100 as distance_from_sma200
        FROM trend_indicators t
        JOIN daily_data d ON t.symbol = d.symbol AND t.date = d.date
        WHERE t.date = (SELECT MAX(date) FROM trend_indicators)
        AND t.sma_200 IS NOT NULL
        AND ABS(d.close_price - t.sma_200) / t.sma_200 * 100 < 2
        ORDER BY distance_from_sma200
        LIMIT 10
    """
    near_support = pd.read_sql_query(query, conn)

    if not near_support.empty:
        print("\nStocks Near 200-Day SMA (Potential Support/Resistance):")
        for _, row in near_support.iterrows():
            print(
                f"  {row['symbol']}: ${row['close_price']:.2f} "
                f"(SMA200: ${row['sma_200']:.2f}, Distance: {row['distance_from_sma200']:.2f}%)"
            )

    conn.close()

    print("\n" + "=" * 60)
    print("INDICATOR CALCULATION COMPLETE")
    print("=" * 60)

    return {
        "total_stocks": int(stats["total_stocks"].iloc[0]) if not stats.empty else 0,
        "total_records": int(stats["total_records"].iloc[0]) if not stats.empty else 0,
        "trend_distribution": (
            trend_dist.to_dict("records") if not trend_dist.empty else []
        ),
        "strong_trends": (
            strong_trends.head(5).to_dict("records") if not strong_trends.empty else []
        ),
    }


if __name__ == "__main__":
    summary = generate_indicator_summary()

    # Save summary to JSON
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports",
        "indicator_summary.json",
    )

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {report_path}")
