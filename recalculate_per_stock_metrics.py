#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新计算每档股票的 Sharpe/Sortino Ratio
避免平均权益曲线导致的虚高问题
"""
import json
import numpy as np
import sys
import io
from pathlib import Path
import glob

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("RECALCULATING PER-STOCK SHARPE & SORTINO RATIOS")
print("=" * 80)

# Load existing metrics
metrics_path = "reports/backtest/local_ppo_oos_full_4215_2023_2025_metrics.json"
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

print(f"\n[1/5] Loading backtest output...")
print(f"Reading from: backtest_full_output.txt")

# Parse backtest output to extract per-stock results
per_stock_results = []
current_stock = None
current_return = None

with open("backtest_full_output.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if "Return=" in line and "Trades=" in line:
            # Example: "  AAL: Return=3.65%, Trades=2"
            parts = line.split(":")
            if len(parts) == 2:
                symbol = parts[0].strip()
                info = parts[1].strip()

                # Extract return
                return_str = info.split(",")[0].replace("Return=", "").replace("%", "").strip()
                try:
                    total_return = float(return_str) / 100.0
                    per_stock_results.append({
                        "symbol": symbol,
                        "total_return": total_return
                    })
                except ValueError:
                    pass

print(f"[OK] Extracted {len(per_stock_results)} stock results")

# Since we don't have per-stock daily equity curves in the output,
# we'll need to load the raw parquet data and re-run the backtest logic
# for each stock to get daily returns. But that would take too long.
#
# Alternative: Estimate per-stock Sharpe from total returns
# Assuming each stock has similar number of trading days (~427 from equity curve length)

print(f"\n[2/5] Calculating per-stock metrics...")

# Get number of trading days from equity curve
n_days = len(metrics["equity_curve"])
print(f"Total trading days: {n_days}")

per_stock_metrics = []

for stock in per_stock_results:
    symbol = stock["symbol"]
    total_return = stock["total_return"]

    # Estimate annualized return
    # total_return = (1 + daily_return)^n_days - 1
    # => daily_return = (1 + total_return)^(1/n_days) - 1
    if total_return > -1:  # Avoid log of negative numbers
        daily_return = (1 + total_return) ** (1/n_days) - 1
    else:
        daily_return = -1.0 / n_days  # Handle extreme losses

    # For Sharpe estimation, we need to estimate volatility
    # Without actual daily returns, we can't calculate exact Sharpe
    # We'll mark this as "N/A - needs daily data"

    per_stock_metrics.append({
        "symbol": symbol,
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (252/n_days) - 1,
        "avg_daily_return": daily_return,
        "sharpe_ratio": None,  # Need actual daily returns
        "sortino_ratio": None  # Need actual daily returns
    })

print(f"[OK] Processed {len(per_stock_metrics)} stocks")

print(f"\n[IMPORTANT] Cannot calculate accurate per-stock Sharpe/Sortino without daily equity curves!")
print(f"We only have:")
print(f"  - Total return per stock")
print(f"  - Average equity curve across all stocks")
print(f"\nTo get accurate per-stock Sharpe, we need to:")
print(f"  1. Store per-stock daily equity curves during backtest")
print(f"  2. Calculate daily returns for each stock")
print(f"  3. Compute Sharpe/Sortino from those daily returns")

print(f"\n[3/5] Computing statistics from available data...")

total_returns = [s["total_return"] for s in per_stock_metrics]
annualized_returns = [s["annualized_return"] for s in per_stock_metrics]

stats = {
    "per_stock_statistics": {
        "note": "Calculated from total returns only. Accurate Sharpe/Sortino requires daily equity curves.",
        "total_return": {
            "mean": float(np.mean(total_returns)),
            "median": float(np.median(total_returns)),
            "std": float(np.std(total_returns)),
            "min": float(np.min(total_returns)),
            "25th": float(np.percentile(total_returns, 25)),
            "75th": float(np.percentile(total_returns, 75)),
            "max": float(np.max(total_returns))
        },
        "annualized_return": {
            "mean": float(np.mean(annualized_returns)),
            "median": float(np.median(annualized_returns)),
            "std": float(np.std(annualized_returns)),
            "min": float(np.min(annualized_returns)),
            "25th": float(np.percentile(annualized_returns, 25)),
            "75th": float(np.percentile(annualized_returns, 75)),
            "max": float(np.max(annualized_returns))
        },
        "sharpe_ratio": {
            "note": "Requires daily returns - not available from current data",
            "method": "Need to store per-stock equity curves during backtest"
        },
        "sortino_ratio": {
            "note": "Requires daily returns - not available from current data",
            "method": "Need to store per-stock equity curves during backtest"
        }
    },
    "equal_weight_portfolio": {
        "note": "Calculated from average equity curve of all stocks",
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sortino_ratio": metrics["sortino_ratio"],
        "warning": "This Sharpe is inflated due to diversification effect (averaging 4,215 stocks)"
    }
}

print(f"\nPer-Stock Total Return Statistics:")
print(f"  Mean: {stats['per_stock_statistics']['total_return']['mean']*100:.2f}%")
print(f"  Median: {stats['per_stock_statistics']['total_return']['median']*100:.2f}%")
print(f"  Std: {stats['per_stock_statistics']['total_return']['std']*100:.2f}%")

print(f"\nPer-Stock Annualized Return Statistics:")
print(f"  Mean: {stats['per_stock_statistics']['annualized_return']['mean']*100:.2f}%")
print(f"  Median: {stats['per_stock_statistics']['annualized_return']['median']*100:.2f}%")
print(f"  Std: {stats['per_stock_statistics']['annualized_return']['std']*100:.2f}%")

print(f"\n[4/5] Updating metrics.json...")
metrics.update(stats)

with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"[OK] Updated: {metrics_path}")

print(f"\n[5/5] Summary:")
print(f"=" * 80)
print(f"COMPLETED: Per-stock statistics added to metrics.json")
print(f"\nKEY FINDINGS:")
print(f"1. Average stock return: {stats['per_stock_statistics']['total_return']['mean']*100:.2f}%")
print(f"2. Median stock return: {stats['per_stock_statistics']['total_return']['median']*100:.2f}%")
print(f"3. Equal-weight portfolio Sharpe: {metrics['sharpe_ratio']:.2f} (INFLATED)")
print(f"4. Per-stock Sharpe: CANNOT CALCULATE without daily equity curves")
print(f"\nRECOMMENDATION:")
print(f"To get accurate per-stock Sharpe/Sortino ratios:")
print(f"  - Modify backtest script to save per-stock daily equity curves")
print(f"  - Re-run calculation with daily return data")
print(f"=" * 80)
