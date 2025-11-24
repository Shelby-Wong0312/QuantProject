#!/usr/bin/env python3
"""
PPO Local Model FULL OOS Backtest (2021-2022) - All 4,215 Stocks
使用本地 parquet 資料對 PPO 模型進行全量樣本外回測
包含視覺化圖表生成
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import matplotlib
matplotlib.use('Agg')  # 使用非GUI後端
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 設置字體支持
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class PPOConfig:
    """PPO配置（與訓練時相同）"""
    def __init__(self):
        self.obs_dim = 220
        self.action_dim = 4
        self.hidden_dim = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """PPO Actor-Critic網路（與訓練時相同）"""
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


class LocalDataLoader:
    """本地Parquet數據載入器"""

    def __init__(self, data_dir: str = "scripts/download/historical_data/daily"):
        self.data_dir = data_dir

    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """載入單個股票的歷史數據"""
        possible_paths = [
            os.path.join(self.data_dir, f"{symbol}_daily.parquet"),
            os.path.join(self.data_dir, f"{symbol.upper()}_daily.parquet"),
            os.path.join(self.data_dir, f"{symbol.lower()}_daily.parquet"),
        ]

        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df = df[(df.index >= start_date) & (df.index <= end_date)]

            df.columns = df.columns.str.lower()
            required_cols = ["open", "high", "low", "close", "volume"]

            if all(col in df.columns for col in required_cols):
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                })
                return df

        except Exception as e:
            pass

        return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵（與訓練時相同）"""
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
    """回測引擎"""

    def __init__(self, model, config: PPOConfig, initial_capital: float = 100000):
        self.model = model
        self.config = config
        self.initial_capital = initial_capital
        self.data_loader = LocalDataLoader()

    def run_single_stock(self, symbol: str, data: pd.DataFrame) -> Dict:
        """對單個股票執行回測"""
        if len(data) < 252:
            return None

        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []

        for i in range(252, len(data)):
            window_data = data.iloc[:i+1]
            current_price = data.iloc[i]["Close"]

            try:
                features = self.data_loader.prepare_features(window_data)
                state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.config.device)
            except Exception:
                equity_curve.append(capital)
                continue

            with torch.no_grad():
                action_logits, _ = self.model(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()

            if action == 1 and position == 0:
                position = (capital * 0.9) / current_price
                entry_price = current_price
                capital -= position * current_price * 1.001
                trades.append({
                    "date": str(data.index[i]),
                    "action": "BUY",
                    "price": current_price,
                    "shares": position,
                })

            elif (action == 2 or action == 3) and position > 0:
                pnl = position * (current_price - entry_price)
                capital += position * current_price * 0.999
                trades.append({
                    "date": str(data.index[i]),
                    "action": "SELL",
                    "price": current_price,
                    "shares": position,
                    "pnl": pnl,
                })
                position = 0
                entry_price = 0

            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)

        if position > 0:
            final_price = data.iloc[-1]["Close"]
            pnl = position * (final_price - entry_price)
            capital += position * final_price * 0.999
            trades.append({
                "date": str(data.index[-1]),
                "action": "SELL",
                "price": final_price,
                "shares": position,
                "pnl": pnl,
            })

        final_equity = capital

        return {
            "symbol": symbol,
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": (final_equity - self.initial_capital) / self.initial_capital,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """計算整體回測指標"""
        all_equity_curves = [r["equity_curve"] for r in results if r and "equity_curve" in r]

        if not all_equity_curves:
            return {}

        min_len = min(len(ec) for ec in all_equity_curves)
        avg_equity = np.mean([ec[:min_len] for ec in all_equity_curves], axis=0)

        returns = np.diff(avg_equity) / avg_equity[:-1]

        cummax = np.maximum.accumulate(avg_equity)
        drawdown = (avg_equity - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # 找到最大回撤的位置
        max_dd_idx = np.argmin(drawdown)

        all_trades = []
        for r in results:
            if r and "trades" in r:
                all_trades.extend([t for t in r["trades"] if t.get("action") == "SELL" and "pnl" in t])

        winning_trades = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        total_trades = len(all_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_returns = [r["total_return"] for r in results if r and "total_return" in r]
        avg_return = np.mean(total_returns) if total_returns else 0
        median_return = np.median(total_returns) if total_returns else 0

        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # 計算Sortino Ratio（只考慮下行波動）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = sharpe

        metrics = {
            "total_stocks": len(results),
            "avg_return": avg_return,
            "median_return": median_return,
            "max_drawdown": max_drawdown,
            "max_drawdown_idx": int(max_dd_idx),
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "equity_curve": avg_equity.tolist(),
            "drawdown_curve": drawdown.tolist(),
            "returns_distribution": {
                "min": float(np.min(total_returns)) if total_returns else 0,
                "25th": float(np.percentile(total_returns, 25)) if total_returns else 0,
                "median": median_return,
                "75th": float(np.percentile(total_returns, 75)) if total_returns else 0,
                "max": float(np.max(total_returns)) if total_returns else 0,
            }
        }

        return metrics


def generate_visualizations(metrics: Dict, results: List[Dict], output_dir: str):
    """生成視覺化圖表"""
    print("\n[VIZ] Generating visualizations...")

    # 創建圖表目錄
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. 權益曲線圖
    plt.figure(figsize=(14, 6))
    equity = np.array(metrics['equity_curve'])
    days = np.arange(len(equity))

    plt.plot(days, equity, linewidth=2, color='#2E86AB', label='Portfolio Equity')
    plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.fill_between(days, 100000, equity, where=(equity >= 100000), alpha=0.2, color='green')
    plt.fill_between(days, 100000, equity, where=(equity < 100000), alpha=0.2, color='red')

    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.title('PPO Model - Portfolio Equity Curve (2021-2022)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    equity_path = os.path.join(viz_dir, "equity_curve.png")
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Equity curve saved: {equity_path}")

    # 2. 回撤圖
    plt.figure(figsize=(14, 6))
    drawdown = np.array(metrics['drawdown_curve']) * 100  # 轉換為百分比

    plt.fill_between(days, 0, drawdown, color='red', alpha=0.3)
    plt.plot(days, drawdown, linewidth=2, color='darkred', label='Drawdown')
    plt.axhline(y=metrics['max_drawdown']*100, color='red', linestyle='--',
                label=f'Max Drawdown: {metrics["max_drawdown"]*100:.2f}%')

    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title('PPO Model - Drawdown Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    dd_path = os.path.join(viz_dir, "drawdown_curve.png")
    plt.savefig(dd_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Drawdown curve saved: {dd_path}")

    # 3. 報酬分佈圖
    returns_data = [r["total_return"] * 100 for r in results if r and "total_return" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 直方圖
    ax1.hist(returns_data, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(x=metrics['avg_return']*100, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {metrics["avg_return"]*100:.2f}%')
    ax1.axvline(x=metrics['median_return']*100, color='green', linestyle='--',
                linewidth=2, label=f'Median: {metrics["median_return"]*100:.2f}%')
    ax1.set_xlabel('Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Returns Distribution Histogram', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 箱形圖
    bp = ax2.boxplot(returns_data, vert=True, patch_artist=True,
                     labels=['All Stocks'])
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][0].set_alpha(0.7)

    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.set_title('Returns Distribution Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    returns_path = os.path.join(viz_dir, "returns_distribution.png")
    plt.savefig(returns_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Returns distribution saved: {returns_path}")

    # 4. 績效摘要圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 勝率餅圖
    win_data = [metrics['winning_trades'], metrics['total_trades'] - metrics['winning_trades']]
    colors = ['#2ECC71', '#E74C3C']
    ax1.pie(win_data, labels=['Winning', 'Losing'], autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title(f'Win Rate: {metrics["win_rate"]*100:.1f}%', fontsize=12, fontweight='bold')

    # 關鍵指標條形圖
    indicators = ['Avg Return\n(%)', 'Sharpe\nRatio', 'Win Rate\n(%)', 'Max DD\n(%)']
    values = [
        metrics['avg_return'] * 100,
        metrics['sharpe_ratio'],
        metrics['win_rate'] * 100,
        abs(metrics['max_drawdown']) * 100
    ]
    colors_bars = ['#2E86AB', '#F77F00', '#2ECC71', '#E74C3C']

    bars = ax2.bar(indicators, values, color=colors_bars, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 在條形圖上標註數值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # 報酬百分位數
    dist = metrics['returns_distribution']
    percentiles = ['Min', '25th', 'Median', '75th', 'Max']
    pct_values = [dist['min']*100, dist['25th']*100, dist['median']*100,
                  dist['75th']*100, dist['max']*100]

    ax3.plot(percentiles, pct_values, marker='o', linewidth=2, markersize=8,
             color='#2E86AB')
    ax3.fill_between(range(len(percentiles)), pct_values, alpha=0.3, color='#2E86AB')
    ax3.set_ylabel('Return (%)', fontsize=10)
    ax3.set_title('Return Percentiles', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 統計摘要文字
    ax4.axis('off')
    summary_text = f"""
    BACKTEST SUMMARY
    {'='*40}

    Total Stocks Tested: {metrics['total_stocks']:,}

    Performance:
    - Average Return: {metrics['avg_return']*100:.2f}%
    - Median Return: {metrics['median_return']*100:.2f}%
    - Max Drawdown: {metrics['max_drawdown']*100:.2f}%

    Risk-Adjusted:
    - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    - Sortino Ratio: {metrics['sortino_ratio']:.2f}

    Trading:
    - Total Trades: {metrics['total_trades']:,}
    - Win Rate: {metrics['win_rate']*100:.2f}%
    - Winning Trades: {metrics['winning_trades']:,}

    Final Equity: ${equity[-1]:,.2f}
    Total Gain: ${equity[-1] - 100000:,.2f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    summary_path = os.path.join(viz_dir, "performance_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Performance summary saved: {summary_path}")

    return viz_dir


def run_full_backtest(
    model_path: str,
    start_date: str = "2021-01-01",
    end_date: str = "2022-12-31",
    output_dir: str = "reports/backtest"
):
    """執行全量OOS回測"""
    print("\n" + "=" * 80)
    print("PPO LOCAL MODEL - FULL OOS BACKTEST (ALL 4,215 STOCKS)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 80)

    # 載入模型
    print("\n[1/6] Loading model...")
    config = PPOConfig()
    model = ActorCritic(config).to(config.device)

    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[OK] Model loaded from {model_path}")
    print(f"[OK] Device: {config.device}")

    # 獲取所有股票
    print(f"\n[2/6] Loading ALL stock data from parquet files...")
    data_dir = "scripts/download/historical_data/daily"
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    print(f"[OK] Found {len(parquet_files)} parquet files")

    # 載入數據
    data_loader = LocalDataLoader(data_dir)
    stock_data = {}

    print("[INFO] Loading data for all stocks (this may take a while)...")
    for file_path in tqdm(parquet_files, desc="Loading parquet files"):
        symbol = os.path.basename(file_path).replace("_daily.parquet", "")
        data = data_loader.load_stock_data(symbol, start_date, end_date)
        if len(data) > 252:
            stock_data[symbol] = data

    print(f"[OK] Successfully loaded {len(stock_data)} stocks with sufficient data")

    # 執行回測
    print(f"\n[3/6] Running backtest on {len(stock_data)} stocks...")
    engine = BacktestEngine(model, config)
    results = []

    for symbol, data in tqdm(stock_data.items(), desc="Backtesting stocks"):
        result = engine.run_single_stock(symbol, data)
        if result:
            results.append(result)

    print(f"[OK] Completed {len(results)} backtests")

    # 計算指標
    print(f"\n[4/6] Computing metrics...")
    metrics = engine.calculate_metrics(results)

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Stocks: {metrics['total_stocks']}")
    print(f"Average Return: {metrics['avg_return']:.2%}")
    print(f"Median Return: {metrics['median_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print("=" * 80)

    # 生成視覺化
    print(f"\n[5/6] Generating visualizations...")
    viz_dir = generate_visualizations(metrics, results, output_dir)

    # 保存結果
    print(f"\n[6/6] Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    # 保存Markdown報告
    report_path = os.path.join(output_dir, "local_ppo_oos_full_4215_2023_2025.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PPO Local Model - FULL OOS Backtest Report (2021-2022)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** `{model_path}`\n\n")
        f.write(f"**Test Period:** {start_date} to {end_date}\n\n")
        f.write(f"**Total Stocks:** {metrics['total_stocks']:,} (All available stocks)\n\n")
        f.write(f"**Data Source:** Local Parquet files (scripts/download/historical_data/daily/)\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Stocks Tested:** {metrics['total_stocks']:,}\n")
        f.write(f"- **Average Return:** {metrics['avg_return']:.2%}\n")
        f.write(f"- **Median Return:** {metrics['median_return']:.2%}\n")
        f.write(f"- **Max Drawdown:** {metrics['max_drawdown']:.2%}\n")
        f.write(f"- **Win Rate:** {metrics['win_rate']:.2%}\n")
        f.write(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Sortino Ratio:** {metrics['sortino_ratio']:.2f}\n")
        f.write(f"- **Total Trades:** {metrics['total_trades']:,}\n")
        f.write(f"- **Winning Trades:** {metrics['winning_trades']:,}\n\n")

        f.write("## Returns Distribution\n\n")
        dist = metrics['returns_distribution']
        f.write("| Percentile | Return |\n")
        f.write("|------------|--------|\n")
        f.write(f"| Minimum | {dist['min']:.2%} |\n")
        f.write(f"| 25th Percentile | {dist['25th']:.2%} |\n")
        f.write(f"| Median (50th) | {dist['median']:.2%} |\n")
        f.write(f"| 75th Percentile | {dist['75th']:.2%} |\n")
        f.write(f"| Maximum | {dist['max']:.2%} |\n\n")

        f.write("## Top 50 Performers\n\n")
        f.write("| Rank | Symbol | Total Return | Trades | Final Equity |\n")
        f.write("|------|--------|--------------|--------|---------------|\n")
        sorted_results = sorted(results, key=lambda x: x["total_return"], reverse=True)
        for idx, r in enumerate(sorted_results[:50], 1):
            f.write(f"| {idx} | {r['symbol']} | {r['total_return']:.2%} | {len(r['trades'])} | ${r['final_equity']:,.2f} |\n")

        f.write("\n## Bottom 20 Performers\n\n")
        f.write("| Rank | Symbol | Total Return | Trades | Final Equity |\n")
        f.write("|------|--------|--------------|--------|---------------|\n")
        for idx, r in enumerate(sorted_results[-20:], 1):
            f.write(f"| {idx} | {r['symbol']} | {r['total_return']:.2%} | {len(r['trades'])} | ${r['final_equity']:,.2f} |\n")

        f.write("\n## Visualizations\n\n")
        f.write("### Equity Curve\n\n")
        f.write(f"![Equity Curve](visualizations/equity_curve.png)\n\n")
        f.write("### Drawdown Chart\n\n")
        f.write(f"![Drawdown](visualizations/drawdown_curve.png)\n\n")
        f.write("### Returns Distribution\n\n")
        f.write(f"![Returns Distribution](visualizations/returns_distribution.png)\n\n")
        f.write("### Performance Summary\n\n")
        f.write(f"![Performance Summary](visualizations/performance_summary.png)\n\n")

        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        f.write(f"model_path: {model_path}\n")
        f.write(f"start_date: {start_date}\n")
        f.write(f"end_date: {end_date}\n")
        f.write(f"total_stocks: {metrics['total_stocks']}\n")
        f.write(f"initial_capital: 100000\n")
        f.write(f"data_source: Local Parquet (no yfinance)\n")
        f.write("```\n\n")

        f.write("---\n\n")
        f.write("*Report generated by backtest_ppo_full.py*\n")

    print(f"[OK] Report saved to: {report_path}")

    # 保存JSON數據
    json_path = os.path.join(output_dir, "local_ppo_oos_full_4215_2023_2025_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved to: {json_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] FULL BACKTEST COMPLETE!")
    print("=" * 80)

    return results, metrics


def main():
    """主函數"""
    model_path = "models/ppo_local/ppo_model_20251119_115916.pt"
    start_date = "2021-01-01"
    end_date = "2022-12-31"

    results, metrics = run_full_backtest(
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
    )

    return results, metrics


if __name__ == "__main__":
    main()
