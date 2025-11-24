#!/usr/bin/env python3
"""
PPO Local Model OOS Backtest (2023-2025)
使用本地 parquet 資料對 PPO 模型進行樣本外回測
不依賴 yfinance
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
warnings.filterwarnings("ignore")


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
        # 尋找parquet文件
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

            # 處理日期索引
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 過濾日期範圍
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # 標準化列名
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
            print(f"Error loading {symbol}: {e}")

        return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵（與訓練時相同）"""
        features = []

        # 1. 價格特徵 (50維)
        returns = data["Close"].pct_change().fillna(0).values[-50:]
        features.append(returns.flatten() if len(returns.shape) > 1 else returns)

        # 2. 成交量特徵 (20維)
        volume_ma = data["Volume"].rolling(20).mean()
        volume_ratio = (data["Volume"] / volume_ma).fillna(1).values[-20:]
        features.append(volume_ratio.flatten() if len(volume_ratio.shape) > 1 else volume_ratio)

        # 3. 技術指標 (150維)
        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).fillna(50).values[-30:]
        features.append(rsi.flatten() if len(rsi.shape) > 1 else rsi)

        # MACD
        ema12 = data["Close"].ewm(span=12).mean()
        ema26 = data["Close"].ewm(span=26).mean()
        macd = (ema12 - ema26).fillna(0).values[-30:]
        features.append(macd.flatten() if len(macd.shape) > 1 else macd)

        # Bollinger Bands
        sma = data["Close"].rolling(20).mean()
        std = data["Close"].rolling(20).std()
        bb_upper = ((data["Close"] - sma) / (std + 1e-10)).fillna(0).values[-30:]
        features.append(bb_upper.flatten() if len(bb_upper.shape) > 1 else bb_upper)

        # ATR
        high_low = data["High"] - data["Low"]
        high_close = abs(data["High"] - data["Close"].shift())
        low_close = abs(data["Low"] - data["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().fillna(0).values[-30:]
        features.append(atr.flatten() if len(atr.shape) > 1 else atr)

        # 合併所有特徵
        all_features = np.concatenate(features)

        # 確保特徵維度為220
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
        if len(data) < 252:  # 需要足夠的歷史數據
            return None

        # 初始化
        capital = self.initial_capital
        position = 0  # 0=無持倉, >0=持倉數量
        entry_price = 0
        trades = []
        equity_curve = []

        # 從第252天開始（需要歷史數據計算特徵）
        for i in range(252, len(data)):
            window_data = data.iloc[:i+1]
            current_price = data.iloc[i]["Close"]

            # 準備特徵
            try:
                features = self.data_loader.prepare_features(window_data)
                state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.config.device)
            except Exception as e:
                equity_curve.append(capital)
                continue

            # 模型推理
            with torch.no_grad():
                action_logits, _ = self.model(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()

            # 執行交易
            # action: 0=hold, 1=buy, 2=sell, 3=close
            if action == 1 and position == 0:  # Buy signal
                # 使用90%資金買入
                position = (capital * 0.9) / current_price
                entry_price = current_price
                capital -= position * current_price * 1.001  # 含手續費
                trades.append({
                    "date": str(data.index[i]),
                    "action": "BUY",
                    "price": current_price,
                    "shares": position,
                })

            elif (action == 2 or action == 3) and position > 0:  # Sell signal
                # 賣出全部持倉
                pnl = position * (current_price - entry_price)
                capital += position * current_price * 0.999  # 扣手續費
                trades.append({
                    "date": str(data.index[i]),
                    "action": "SELL",
                    "price": current_price,
                    "shares": position,
                    "pnl": pnl,
                })
                position = 0
                entry_price = 0

            # 計算當前權益
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)

        # 最終平倉
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
        # 聚合所有股票的權益曲線
        all_equity_curves = [r["equity_curve"] for r in results if r and "equity_curve" in r]

        if not all_equity_curves:
            return {}

        # 計算平均權益曲線
        min_len = min(len(ec) for ec in all_equity_curves)
        avg_equity = np.mean([ec[:min_len] for ec in all_equity_curves], axis=0)

        # 計算回報
        returns = np.diff(avg_equity) / avg_equity[:-1]

        # 計算最大回撤
        cummax = np.maximum.accumulate(avg_equity)
        drawdown = (avg_equity - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # 計算勝率
        all_trades = []
        for r in results:
            if r and "trades" in r:
                all_trades.extend([t for t in r["trades"] if t.get("action") == "SELL" and "pnl" in t])

        winning_trades = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        total_trades = len(all_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 計算總回報
        total_returns = [r["total_return"] for r in results if r and "total_return" in r]
        avg_return = np.mean(total_returns) if total_returns else 0

        # 計算 Sharpe Ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # 年化
        else:
            sharpe = 0

        metrics = {
            "total_stocks": len(results),
            "avg_return": avg_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "sharpe_ratio": sharpe,
            "equity_curve": avg_equity.tolist(),
        }

        return metrics


def run_oos_backtest(
    model_path: str,
    start_date: str = "2023-01-01",
    end_date: str = "2025-08-08",
    num_stocks: int = 100,
    output_dir: str = "reports/backtest"
):
    """執行OOS回測"""
    print("\n" + "=" * 80)
    print("PPO LOCAL MODEL - OOS BACKTEST (2023-2025)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Stocks: {num_stocks}")
    print("=" * 80)

    # 載入模型
    print("\n[1/5] Loading model...")
    config = PPOConfig()
    model = ActorCritic(config).to(config.device)

    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[OK] Model loaded from {model_path}")
    print(f"[OK] Device: {config.device}")

    # 獲取股票列表
    print(f"\n[2/5] Loading stock data from parquet files...")
    data_dir = "scripts/download/historical_data/daily"
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

    # 選取前N個股票
    selected_files = parquet_files[:num_stocks]
    print(f"[OK] Found {len(parquet_files)} stocks, selected {len(selected_files)}")

    # 載入數據
    data_loader = LocalDataLoader(data_dir)
    stock_data = {}

    for file_path in selected_files:
        symbol = os.path.basename(file_path).replace("_daily.parquet", "")
        data = data_loader.load_stock_data(symbol, start_date, end_date)
        if len(data) > 252:  # 至少需要一年歷史數據
            stock_data[symbol] = data

    print(f"[OK] Successfully loaded {len(stock_data)} stocks with sufficient data")

    # 執行回測
    print(f"\n[3/5] Running backtest on {len(stock_data)} stocks...")
    engine = BacktestEngine(model, config)
    results = []

    for symbol, data in stock_data.items():
        result = engine.run_single_stock(symbol, data)
        if result:
            results.append(result)
            print(f"  {symbol}: Return={result['total_return']:.2%}, Trades={len(result['trades'])}")

    print(f"\n[OK] Completed {len(results)} backtests")

    # 計算指標
    print(f"\n[4/5] Computing metrics...")
    metrics = engine.calculate_metrics(results)

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Stocks: {metrics['total_stocks']}")
    print(f"Average Return: {metrics['avg_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print("=" * 80)

    # 保存結果
    print(f"\n[5/5] Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    # 保存Markdown報告
    report_path = os.path.join(output_dir, "local_ppo_oos_2023_2025.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PPO Local Model - OOS Backtest Report (2023-2025)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** `{model_path}`\n\n")
        f.write(f"**Test Period:** {start_date} to {end_date}\n\n")
        f.write(f"**Data Source:** Local Parquet files (scripts/download/historical_data/daily/)\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Stocks Tested:** {metrics['total_stocks']}\n")
        f.write(f"- **Average Return:** {metrics['avg_return']:.2%}\n")
        f.write(f"- **Max Drawdown:** {metrics['max_drawdown']:.2%}\n")
        f.write(f"- **Win Rate:** {metrics['win_rate']:.2%}\n")
        f.write(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Total Trades:** {metrics['total_trades']}\n")
        f.write(f"- **Winning Trades:** {metrics['winning_trades']}\n\n")

        f.write("## Performance by Stock\n\n")
        f.write("| Symbol | Total Return | Trades | Final Equity |\n")
        f.write("|--------|--------------|--------|---------------|\n")
        for r in sorted(results, key=lambda x: x["total_return"], reverse=True)[:20]:
            f.write(f"| {r['symbol']} | {r['total_return']:.2%} | {len(r['trades'])} | ${r['final_equity']:,.2f} |\n")

        f.write("\n## Equity Curve\n\n")
        f.write("Average equity curve across all stocks:\n\n")
        f.write("```\n")
        equity = metrics['equity_curve']
        step = max(1, len(equity) // 20)
        for i in range(0, len(equity), step):
            f.write(f"Day {i:4d}: ${equity[i]:,.2f}\n")
        f.write("```\n\n")

        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        f.write(f"model_path: {model_path}\n")
        f.write(f"start_date: {start_date}\n")
        f.write(f"end_date: {end_date}\n")
        f.write(f"num_stocks: {num_stocks}\n")
        f.write(f"initial_capital: 100000\n")
        f.write(f"data_source: Local Parquet (no yfinance)\n")
        f.write("```\n\n")

        f.write("---\n\n")
        f.write("*Report generated by backtest_ppo_local_oos.py*\n")

    print(f"[OK] Report saved to: {report_path}")

    # 保存JSON數據
    json_path = os.path.join(output_dir, "local_ppo_oos_2023_2025_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved to: {json_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] BACKTEST COMPLETE!")
    print("=" * 80)

    return results, metrics


def main():
    """主函數"""
    # 配置
    model_path = "models/ppo_local/ppo_model_20251119_115916.pt"
    start_date = "2023-01-01"
    end_date = "2025-08-08"
    num_stocks = 100

    # 執行回測
    results, metrics = run_oos_backtest(
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
        num_stocks=num_stocks,
    )

    return results, metrics


if __name__ == "__main__":
    main()
