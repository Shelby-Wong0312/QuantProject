#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每档股票的 Sharpe/Sortino Ratio
复用 backtest_ppo_full.py 的架构，但保存每档股票的日收益率
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===== 从 backtest_ppo_full.py 复制的类定义 =====

class PPOConfig:
    """PPO配置"""
    def __init__(self):
        self.obs_dim = 220
        self.action_dim = 4
        self.hidden_dim = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """PPO Actor-Critic网络（完全匹配 backtest_ppo_full.py）"""
    def __init__(self, config: PPOConfig):
        super(ActorCritic, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value


class DataLoaderSimple:
    """本地数据加载器（匹配 backtest_ppo_full.py）"""
    def __init__(self):
        pass

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备220维特征（与训练时相同）"""
        features = []

        returns = data["Close"].pct_change().fillna(0).values[-50:]
        features.append(returns.flatten() if len(returns.shape) > 1 else returns)

        volume_ma = data["Volume"].rolling(20).mean()
        volume_ratio = (data["Volume"] / volume_ma).fillna(1).values[-20:]
        features.append(volume_ratio.flatten() if len(volume_ratio.shape) > 1 else volume_ratio)

        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).fillna(50).values[-30:]
        features.append(rsi.flatten() if len(rsi.shape) > 1 else rsi)

        ema12 = data["Close"].ewm(span=12).mean()
        ema26 = data["Close"].ewm(span=26).mean()
        macd = (ema12 - ema26).fillna(0).values[-30:]
        features.append(macd.flatten() if len(macd.shape) > 1 else macd)

        sma = data["Close"].rolling(20).mean()
        std = data["Close"].rolling(20).std()
        bb_upper = ((data["Close"] - sma) / (std + 1e-10)).fillna(0).values[-30:]
        features.append(bb_upper.flatten() if len(bb_upper.shape) > 1 else bb_upper)

        high_low = data["High"] - data["Low"]
        high_close = abs(data["High"] - data["Close"].shift())
        low_close = abs(data["Low"] - data["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().fillna(0).values[-30:]
        features.append(atr.flatten() if len(atr.shape) > 1 else atr)

        all_features = np.concatenate(features)

        if len(all_features) < 220:
            all_features = np.pad(all_features, (0, 220 - len(all_features)), "constant")
        elif len(all_features) > 220:
            all_features = all_features[:220]

        return all_features


class BacktestEngine:
    """回测引擎"""
    def __init__(self, model: ActorCritic, config: PPOConfig):
        self.model = model
        self.config = config
        self.data_loader = DataLoaderSimple()
        self.initial_capital = 100000

    def backtest_single_stock(self, symbol: str, data: pd.DataFrame) -> Dict:
        """对单一股票进行回测，返回日收益率"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        equity_curve = []

        for i in range(252, len(data)):
            window_data = data.iloc[:i+1]
            current_price = data.iloc[i]["Close"]

            try:
                features = self.data_loader.prepare_features(window_data)
                state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.config.device)
            except Exception:
                equity_curve.append(capital + (position * current_price if position > 0 else 0))
                continue

            with torch.no_grad():
                action_logits, _ = self.model(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()

            if action == 1 and position == 0:
                position = (capital * 0.9) / current_price
                entry_price = current_price
                capital -= position * current_price * 1.001

            elif (action == 2 or action == 3) and position > 0:
                capital += position * current_price * 0.999
                position = 0
                entry_price = 0

            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)

        if position > 0:
            final_price = data.iloc[-1]["Close"]
            capital += position * final_price * 0.999

        final_equity = capital

        # 计算指标
        if len(equity_curve) < 2:
            return None

        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        total_return = (final_equity - self.initial_capital) / self.initial_capital

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

        return {
            "symbol": symbol,
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "n_days": len(equity_curve)
        }


def main():
    print("=" * 80)
    print("CALCULATING PER-STOCK SHARPE & SORTINO RATIOS")
    print("=" * 80)

    # Configuration
    model_path = "models/ppo_local/ppo_model_20251119_115916.pt"
    data_dir = "scripts/download/historical_data/daily"
    start_date = "2023-01-01"
    end_date = "2025-08-08"

    print(f"\n[1/6] Loading configuration...")
    config = PPOConfig()
    print(f"[OK] Device: {config.device}")

    print(f"\n[2/6] Loading model...")
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model = ActorCritic(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[OK] Model loaded from {model_path}")

    print(f"\n[3/6] Loading data...")
    parquet_files = list(Path(data_dir).glob("*.parquet"))
    print(f"[OK] Found {len(parquet_files)} parquet files")

    print(f"\n[4/6] Calculating per-stock Sharpe/Sortino...")
    print("This will take approximately 2 hours...")

    engine = BacktestEngine(model, config)
    per_stock_results = []

    for pf in tqdm(parquet_files, desc="Processing stocks"):
        symbol = pf.stem.replace("_daily", "")

        try:
            df = pd.read_parquet(pf)

            # Fix index and columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            # Capitalize column names
            df.columns = [c.capitalize() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]

            # Filter by date
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if len(df) < 300:
                continue

            result = engine.backtest_single_stock(symbol, df)
            if result:
                per_stock_results.append(result)

        except Exception:
            continue

    print(f"\n[OK] Processed {len(per_stock_results)} stocks successfully")

    print(f"\n[5/6] Computing statistics...")

    if len(per_stock_results) == 0:
        print("[ERROR] No results to analyze!")
        sys.exit(1)

    # Extract metrics
    total_returns = [r["total_return"] for r in per_stock_results]
    sharpe_ratios = [r["sharpe_ratio"] for r in per_stock_results]
    sortino_ratios = [r["sortino_ratio"] for r in per_stock_results]
    max_drawdowns = [r["max_drawdown"] for r in per_stock_results]

    # Compute statistics
    stats = {
        "per_stock_statistics": {
            "note": "Calculated from individual stock daily returns",
            "n_stocks": len(per_stock_results),
            "total_return": {
                "mean": float(np.mean(total_returns)),
                "median": float(np.median(total_returns)),
                "std": float(np.std(total_returns)),
                "min": float(np.min(total_returns)),
                "25th": float(np.percentile(total_returns, 25)),
                "75th": float(np.percentile(total_returns, 75)),
                "max": float(np.max(total_returns))
            },
            "sharpe_ratio": {
                "mean": float(np.mean(sharpe_ratios)),
                "median": float(np.median(sharpe_ratios)),
                "std": float(np.std(sharpe_ratios)),
                "min": float(np.min(sharpe_ratios)),
                "25th": float(np.percentile(sharpe_ratios, 25)),
                "75th": float(np.percentile(sharpe_ratios, 75)),
                "max": float(np.max(sharpe_ratios))
            },
            "sortino_ratio": {
                "mean": float(np.mean(sortino_ratios)),
                "median": float(np.median(sortino_ratios)),
                "std": float(np.std(sortino_ratios)),
                "min": float(np.min(sortino_ratios)),
                "25th": float(np.percentile(sortino_ratios, 25)),
                "75th": float(np.percentile(sortino_ratios, 75)),
                "max": float(np.max(sortino_ratios))
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)),
                "median": float(np.median(max_drawdowns)),
                "std": float(np.std(max_drawdowns)),
                "min": float(np.min(max_drawdowns)),
                "25th": float(np.percentile(max_drawdowns, 25)),
                "75th": float(np.percentile(max_drawdowns, 75)),
                "max": float(np.max(max_drawdowns))
            }
        },
        "top_10_by_sharpe": sorted(per_stock_results, key=lambda x: x["sharpe_ratio"], reverse=True)[:10],
        "bottom_10_by_sharpe": sorted(per_stock_results, key=lambda x: x["sharpe_ratio"])[:10]
    }

    # Print results
    print(f"\n{'='*80}")
    print("PER-STOCK STATISTICS (N={})".format(stats['per_stock_statistics']['n_stocks']))
    print('='*80)

    print(f"\nSharpe Ratio:")
    print(f"  Mean:   {stats['per_stock_statistics']['sharpe_ratio']['mean']:>8.3f}")
    print(f"  Median: {stats['per_stock_statistics']['sharpe_ratio']['median']:>8.3f}")
    print(f"  Std:    {stats['per_stock_statistics']['sharpe_ratio']['std']:>8.3f}")
    print(f"  Range:  [{stats['per_stock_statistics']['sharpe_ratio']['min']:.2f}, {stats['per_stock_statistics']['sharpe_ratio']['max']:.2f}]")

    print(f"\nSortino Ratio:")
    print(f"  Mean:   {stats['per_stock_statistics']['sortino_ratio']['mean']:>8.3f}")
    print(f"  Median: {stats['per_stock_statistics']['sortino_ratio']['median']:>8.3f}")

    print(f"\nTotal Return:")
    print(f"  Mean:   {stats['per_stock_statistics']['total_return']['mean']*100:>8.2f}%")
    print(f"  Median: {stats['per_stock_statistics']['total_return']['median']*100:>8.2f}%")

    print(f"\nMax Drawdown:")
    print(f"  Mean:   {stats['per_stock_statistics']['max_drawdown']['mean']*100:>8.2f}%")
    print(f"  Median: {stats['per_stock_statistics']['max_drawdown']['median']*100:>8.2f}%")

    print(f"\n[6/6] Saving results...")

    # Load existing metrics and update
    metrics_path = "reports/backtest/local_ppo_oos_full_4215_2023_2025_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Add per-stock statistics
    metrics["per_stock_statistics"] = stats["per_stock_statistics"]
    metrics["equal_weight_portfolio_sharpe"] = {
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sortino_ratio": metrics["sortino_ratio"],
        "note": "Calculated from AVERAGE equity curve across all stocks",
        "warning": "Inflated by diversification effect (~sqrt(N) where N=4215)"
    }
    metrics["top_10_stocks_by_sharpe"] = stats["top_10_by_sharpe"]
    metrics["bottom_10_stocks_by_sharpe"] = stats["bottom_10_by_sharpe"]

    # Save individual stock details
    details_path = "reports/backtest/local_ppo_oos_full_4215_2023_2025_per_stock.json"
    with open(details_path, 'w') as f:
        json.dump(per_stock_results, f, indent=2)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Updated: {metrics_path}")
    print(f"[OK] Saved detailed results: {details_path}")

    print(f"\n{'='*80}")
    print("COMPLETED!")
    print('='*80)
    print(f"\nKEY FINDINGS:")
    print(f"1. Per-Stock Sharpe (Mean):     {stats['per_stock_statistics']['sharpe_ratio']['mean']:.3f}")
    print(f"2. Per-Stock Sharpe (Median):   {stats['per_stock_statistics']['sharpe_ratio']['median']:.3f}")
    print(f"3. Portfolio Sharpe (Inflated): {metrics['sharpe_ratio']:.2f}")
    print(f"4. Inflation Factor:            ~{metrics['sharpe_ratio'] / max(stats['per_stock_statistics']['sharpe_ratio']['mean'], 0.01):.1f}x")
    print('='*80)


if __name__ == "__main__":
    main()
