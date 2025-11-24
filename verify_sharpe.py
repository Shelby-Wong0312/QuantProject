#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import sys
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Read metrics
with open('reports/backtest/local_ppo_oos_full_4215_2023_2025_metrics.json', 'r') as f:
    data = json.load(f)

equity = np.array(data['equity_curve'])
returns = np.diff(equity) / equity[:-1]

print("=" * 80)
print("SHARPE RATIO VERIFICATION")
print("=" * 80)
print(f"\nTotal data points: {len(equity)}")
print(f"Returns series length: {len(returns)}")
print(f"\nMean daily return: {np.mean(returns)*100:.6f}%")
print(f"Std of returns: {np.std(returns)*100:.6f}%")
print(f"\nSharpe calculation:")
print(f"  = (mean daily return / std) * sqrt(252)")
print(f"  = ({np.mean(returns):.8f} / {np.std(returns):.8f}) * {np.sqrt(252):.4f}")
print(f"  = {np.mean(returns) / np.std(returns):.4f} * {np.sqrt(252):.4f}")
print(f"  = {np.mean(returns) / np.std(returns) * np.sqrt(252):.2f}")

print(f"\n\nPROBLEM IDENTIFIED:")
print(f"This Sharpe is calculated on the AVERAGE equity curve of 4,215 stocks!")
print(f"Due to averaging effect, volatility is severely compressed.")
print(f"\nCorrect approach:")
print(f"1. Calculate Sharpe for each stock individually")
print(f"2. Report mean/median Sharpe across stocks")
print("=" * 80)
