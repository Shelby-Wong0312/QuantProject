#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buy-and-Hold Benchmark Backtest (2023-2025)
对同样的 4,215 档股票执行买入持有策略，作为 PPO 策略的基准对比
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

print("=" * 80)
print("BUY-AND-HOLD BENCHMARK BACKTEST (2023-2025)")
print("=" * 80)

# Configuration
data_dir = "scripts/download/historical_data/daily"
start_date = "2023-01-01"
end_date = "2025-08-08"
initial_capital = 100000
transaction_cost = 0.001  # 0.1% to match PPO backtest

print(f"\n[1/5] Configuration...")
print(f"Data directory: {data_dir}")
print(f"Period: {start_date} to {end_date}")
print(f"Initial capital: ${initial_capital:,.0f}")
print(f"Transaction cost: {transaction_cost*100:.1f}%")

print(f"\n[2/5] Loading parquet files...")
parquet_files = list(Path(data_dir).glob("*.parquet"))
print(f"[OK] Found {len(parquet_files)} stocks")

print(f"\n[3/5] Running buy-and-hold backtest...")
print("This will take approximately 5-10 minutes...")

buy_hold_results = []

for pf in tqdm(parquet_files, desc="Processing stocks"):
    symbol = pf.stem.replace("_daily", "")

    try:
        # Load data
        df = pd.read_parquet(pf)

        # Fix index and columns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Capitalize column names
        df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]

        # Filter by date
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if len(df) < 2:  # Need at least 2 data points
            continue

        # Buy-and-hold strategy
        buy_price = df.iloc[0]['Close']
        sell_price = df.iloc[-1]['Close']

        # Account for transaction costs
        shares = (initial_capital * (1 - transaction_cost)) / buy_price
        final_capital = shares * sell_price * (1 - transaction_cost)

        total_return = (final_capital - initial_capital) / initial_capital

        # Calculate daily equity curve for Sharpe calculation
        equity_curve = []
        for i in range(len(df)):
            current_price = df.iloc[i]['Close']
            current_equity = shares * current_price
            equity_curve.append(current_equity)

        equity_curve = np.array(equity_curve)

        # Calculate daily returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Sharpe Ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max Drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)

        buy_hold_results.append({
            "symbol": symbol,
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "n_days": len(df),
            "buy_price": float(buy_price),
            "sell_price": float(sell_price)
        })

    except Exception as e:
        continue

print(f"\n[OK] Processed {len(buy_hold_results)} stocks successfully")

print(f"\n[4/5] Computing statistics...")

if len(buy_hold_results) == 0:
    print("[ERROR] No results to analyze!")
    sys.exit(1)

# Extract metrics
bh_total_returns = [r["total_return"] for r in buy_hold_results]
bh_sharpe_ratios = [r["sharpe_ratio"] for r in buy_hold_results]
bh_sortino_ratios = [r["sortino_ratio"] for r in buy_hold_results]
bh_max_drawdowns = [r["max_drawdown"] for r in buy_hold_results]

# Compute statistics
bh_stats = {
    "buy_and_hold_benchmark": {
        "note": "Simple buy-and-hold strategy for comparison",
        "n_stocks": len(buy_hold_results),
        "period": f"{start_date} to {end_date}",
        "total_return": {
            "mean": float(np.mean(bh_total_returns)),
            "median": float(np.median(bh_total_returns)),
            "std": float(np.std(bh_total_returns)),
            "min": float(np.min(bh_total_returns)),
            "25th": float(np.percentile(bh_total_returns, 25)),
            "75th": float(np.percentile(bh_total_returns, 75)),
            "max": float(np.max(bh_total_returns))
        },
        "sharpe_ratio": {
            "mean": float(np.mean(bh_sharpe_ratios)),
            "median": float(np.median(bh_sharpe_ratios)),
            "std": float(np.std(bh_sharpe_ratios)),
            "min": float(np.min(bh_sharpe_ratios)),
            "25th": float(np.percentile(bh_sharpe_ratios, 25)),
            "75th": float(np.percentile(bh_sharpe_ratios, 75)),
            "max": float(np.max(bh_sharpe_ratios))
        },
        "sortino_ratio": {
            "mean": float(np.mean(bh_sortino_ratios)),
            "median": float(np.median(bh_sortino_ratios)),
            "std": float(np.std(bh_sortino_ratios)),
            "min": float(np.min(bh_sortino_ratios)),
            "25th": float(np.percentile(bh_sortino_ratios, 25)),
            "75th": float(np.percentile(bh_sortino_ratios, 75)),
            "max": float(np.max(bh_sortino_ratios))
        },
        "max_drawdown": {
            "mean": float(np.mean(bh_max_drawdowns)),
            "median": float(np.median(bh_max_drawdowns)),
            "std": float(np.std(bh_max_drawdowns)),
            "min": float(np.min(bh_max_drawdowns)),
            "25th": float(np.percentile(bh_max_drawdowns, 25)),
            "75th": float(np.percentile(bh_max_drawdowns, 75)),
            "max": float(np.max(bh_max_drawdowns))
        }
    }
}

# Print results
print(f"\n{'='*80}")
print("BUY-AND-HOLD BENCHMARK STATISTICS (N={})".format(bh_stats['buy_and_hold_benchmark']['n_stocks']))
print('='*80)

print(f"\nTotal Return:")
print(f"  Mean:   {bh_stats['buy_and_hold_benchmark']['total_return']['mean']*100:>8.2f}%")
print(f"  Median: {bh_stats['buy_and_hold_benchmark']['total_return']['median']*100:>8.2f}%")
print(f"  Std:    {bh_stats['buy_and_hold_benchmark']['total_return']['std']*100:>8.2f}%")

print(f"\nSharpe Ratio:")
print(f"  Mean:   {bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['mean']:>8.3f}")
print(f"  Median: {bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['median']:>8.3f}")
print(f"  Std:    {bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['std']:>8.3f}")

print(f"\nSortino Ratio:")
print(f"  Mean:   {bh_stats['buy_and_hold_benchmark']['sortino_ratio']['mean']:>8.3f}")
print(f"  Median: {bh_stats['buy_and_hold_benchmark']['sortino_ratio']['median']:>8.3f}")

print(f"\nMax Drawdown:")
print(f"  Mean:   {bh_stats['buy_and_hold_benchmark']['max_drawdown']['mean']*100:>8.2f}%")
print(f"  Median: {bh_stats['buy_and_hold_benchmark']['max_drawdown']['median']*100:>8.2f}%")

# Load PPO results for comparison
print(f"\n[5/5] Comparing with PPO strategy...")

metrics_path = "reports/backtest/local_ppo_oos_full_4215_2023_2025_metrics.json"
with open(metrics_path, 'r') as f:
    ppo_metrics = json.load(f)

ppo_stats = ppo_metrics['per_stock_statistics']

# Calculate alpha (excess returns)
alpha_return = ppo_stats['total_return']['mean'] - bh_stats['buy_and_hold_benchmark']['total_return']['mean']
alpha_sharpe = ppo_stats['sharpe_ratio']['mean'] - bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['mean']
alpha_sortino = ppo_stats['sortino_ratio']['mean'] - bh_stats['buy_and_hold_benchmark']['sortino_ratio']['mean']

# Calculate win rate (how many stocks did PPO beat B&H)
ppo_per_stock = json.load(open("reports/backtest/local_ppo_oos_full_4215_2023_2025_per_stock.json"))
bh_dict = {r['symbol']: r for r in buy_hold_results}

beats_bh = 0
total_compared = 0

for ppo_stock in ppo_per_stock:
    symbol = ppo_stock['symbol']
    if symbol in bh_dict:
        total_compared += 1
        if ppo_stock['total_return'] > bh_dict[symbol]['total_return']:
            beats_bh += 1

beat_rate = beats_bh / total_compared if total_compared > 0 else 0

comparison_stats = {
    "ppo_vs_buy_hold": {
        "note": "Comparison of PPO strategy vs buy-and-hold benchmark",
        "alpha": {
            "total_return": float(alpha_return),
            "sharpe_ratio": float(alpha_sharpe),
            "sortino_ratio": float(alpha_sortino)
        },
        "beat_rate": float(beat_rate),
        "stocks_compared": total_compared,
        "ppo_beats_bh": beats_bh,
        "relative_performance": {
            "return_ratio": float(ppo_stats['total_return']['mean'] / bh_stats['buy_and_hold_benchmark']['total_return']['mean']) if bh_stats['buy_and_hold_benchmark']['total_return']['mean'] != 0 else 0,
            "sharpe_ratio": float(ppo_stats['sharpe_ratio']['mean'] / bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['mean']) if bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['mean'] != 0 else 0
        }
    }
}

# Print comparison
print(f"\n{'='*80}")
print("PPO STRATEGY vs BUY-AND-HOLD COMPARISON")
print('='*80)

print(f"\n📊 Average Returns:")
print(f"  PPO Strategy:    {ppo_stats['total_return']['mean']*100:>8.2f}%")
print(f"  Buy-and-Hold:    {bh_stats['buy_and_hold_benchmark']['total_return']['mean']*100:>8.2f}%")
print(f"  Alpha (Excess):  {alpha_return*100:>8.2f}%")

print(f"\n📈 Sharpe Ratios:")
print(f"  PPO Strategy:    {ppo_stats['sharpe_ratio']['mean']:>8.3f}")
print(f"  Buy-and-Hold:    {bh_stats['buy_and_hold_benchmark']['sharpe_ratio']['mean']:>8.3f}")
print(f"  Alpha (Excess):  {alpha_sharpe:>8.3f}")

print(f"\n🎯 Win Rate:")
print(f"  PPO beats B&H:   {beats_bh}/{total_compared} stocks ({beat_rate*100:.2f}%)")

print(f"\n📉 Risk Metrics:")
print(f"  PPO Avg Drawdown:  {ppo_stats['max_drawdown']['mean']*100:>8.2f}%")
print(f"  B&H Avg Drawdown:  {bh_stats['buy_and_hold_benchmark']['max_drawdown']['mean']*100:>8.2f}%")

# Save results
print(f"\n[6/6] Saving results...")

# Update metrics.json
ppo_metrics['buy_and_hold_benchmark'] = bh_stats['buy_and_hold_benchmark']
ppo_metrics['ppo_vs_buy_hold'] = comparison_stats['ppo_vs_buy_hold']

with open(metrics_path, 'w') as f:
    json.dump(ppo_metrics, f, indent=2)

# Save detailed B&H results
bh_details_path = "reports/backtest/buy_hold_benchmark_2023_2025.json"
with open(bh_details_path, 'w') as f:
    json.dump(buy_hold_results, f, indent=2)

print(f"[OK] Updated: {metrics_path}")
print(f"[OK] Saved B&H details: {bh_details_path}")

print(f"\n{'='*80}")
print("COMPLETED!")
print('='*80)

print(f"\n🎉 KEY FINDINGS:")
print(f"1. PPO Alpha (Return): {alpha_return*100:+.2f}%")
print(f"2. PPO Alpha (Sharpe): {alpha_sharpe:+.3f}")
print(f"3. PPO Beat Rate: {beat_rate*100:.1f}% of stocks")
print(f"4. Return Ratio: {comparison_stats['ppo_vs_buy_hold']['relative_performance']['return_ratio']:.2f}x")
print('='*80)
