#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO統一監控系統 - 使用訓練好的模型監控所有股票
Unified PPO Monitoring System for All Stocks
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")


class PPONetwork(nn.Module):
    """PPO神經網絡模型"""

    def __init__(self, input_dim=50, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3)  # Buy, Hold, Sell
        )

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        features = self.features(x)
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_probs = action_probs + 1e-8  # 防止log(0)
        value = self.critic(features)
        return action_probs, value


class UnifiedPPOMonitor:
    """統一PPO監控系統"""

    def __init__(self):
        self.model_path = "models/ppo_3488_stocks.pt"
        self.symbols_file = "validated_yahoo_symbols_final.txt"
        self.model = None
        self.symbols = []
        self.monitoring_data = {}
        self.running = False
        self.refresh_interval = 30  # 30秒刷新一次

    def load_model(self):
        """載入PPO模型"""
        print("Loading PPO model...")
        self.model = PPONetwork()

        if os.path.exists(self.model_path):
            checkpoint = torch.load(
                self.model_path, map_location="cpu", weights_only=False
            )
            if "model_state_dict" in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    print("[SUCCESS] Model loaded from checkpoint")
                except Exception:
                    print("[WARNING] Using new model initialization")
        else:
            print("[WARNING] Model file not found, using new initialization")

        self.model.eval()

    def load_symbols(self):
        """載入股票列表"""
        print("Loading stock symbols...")

        if os.path.exists(self.symbols_file):
            with open(self.symbols_file, "r") as f:
                all_symbols = [line.strip() for line in f if line.strip()]
            # 只使用前100個活躍股票進行測試
            self.symbols = all_symbols[:100]
            print(
                f"[SUCCESS] Loaded {len(self.symbols)} symbols (from {len(all_symbols)} total)"
            )
        else:
            # 使用默認測試列表
            self.symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "TSLA",
                "NVDA",
                "META",
                "AMZN",
                "JPM",
                "V",
                "JNJ",
                "WMT",
                "PG",
                "MA",
                "DIS",
                "NVDA",
                "HD",
                "BAC",
                "ADBE",
                "CRM",
                "NFLX",
                "KO",
                "PFE",
                "TMO",
                "CSCO",
                "PEP",
                "AVGO",
                "INTC",
                "CMCSA",
                "VZ",
                "ABBV",
            ]
            print(f"[WARNING] Using default {len(self.symbols)} test symbols")

    def extract_features(self, ticker_data: pd.DataFrame) -> np.ndarray:
        """提取技術特徵"""
        if len(ticker_data) < 30:
            return np.zeros(50)

        features = []
        prices = ticker_data["Close"].values
        volumes = ticker_data["Volume"].values
        highs = ticker_data["High"].values
        lows = ticker_data["Low"].values

        # 價格變化
        returns = np.diff(prices) / prices[:-1]
        features.extend(
            [
                np.mean(returns[-20:]),
                np.std(returns[-20:]),
                (prices[-1] - prices[-20]) / prices[-20],
                (prices[-1] - np.min(prices[-20:]))
                / (np.max(prices[-20:]) - np.min(prices[-20:]) + 1e-8),
            ]
        )

        # 移動平均
        for period in [5, 10, 20]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features.append((prices[-1] / ma) - 1)
            else:
                features.append(0)

        # 成交量
        if len(volumes) > 0:
            features.append(np.mean(volumes[-20:]) / (np.max(volumes[-20:]) + 1e-8))
            features.append(volumes[-1] / (np.mean(volumes[-20:]) + 1e-8))
        else:
            features.extend([0, 0])

        # RSI
        if len(returns) >= 14:
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            if len(gains) > 0 and len(losses) > 0:
                rs = np.mean(gains) / (np.mean(losses) + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100)
            else:
                features.append(0.5)
        else:
            features.append(0.5)

        # 波動率
        features.append(np.std(returns[-20:]) if len(returns) >= 20 else 0)

        # ATR
        if len(ticker_data) >= 14:
            tr = np.maximum(
                highs[-14:] - lows[-14:], np.abs(highs[-14:] - prices[-15:-1])
            )
            atr = np.mean(tr)
            features.append(atr / prices[-1] if prices[-1] > 0 else 0)
        else:
            features.append(0)

        # 補齊到50維
        while len(features) < 50:
            features.append(0)

        return np.array(features[:50], dtype=np.float32)

    def get_signal(self, symbol: str) -> Dict:
        """獲取單個股票的交易信號"""
        try:
            # 下載數據
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            ticker.history(start=start_date, end=end_date)

            if data.empty or len(data) < 30:
                return {
                    "symbol": symbol,
                    "signal": "NO_DATA",
                    "confidence": 0,
                    "price": 0,
                    "change": 0,
                    "error": "Insufficient data",
                }

            # 提取特徵
            features = self.extract_features(data)
            feature_tensor = torch.FloatTensor(features).unsqueeze(0)

            # 獲取預測
            with torch.no_grad():
                action_probs, value = self.model(feature_tensor)
                action_probs = action_probs.squeeze().numpy()

            # 決定信號
            best_action = np.argmax(action_probs)
            confidence = float(action_probs[best_action])

            action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}
            signal = action_map[best_action]

            # 獲取當前價格
            current_price = float(data["Close"].iloc[-1])

            # 計算變化
            price_change = float(
                (data["Close"].iloc[-1] - data["Close"].iloc[-2])
                / data["Close"].iloc[-2]
                * 100
            )

            return {
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "price": current_price,
                "change": price_change,
                "value": float(value.item()),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "signal": "ERROR",
                "confidence": 0,
                "price": 0,
                "change": 0,
                "error": str(e)[:50],  # 限制錯誤信息長度
            }

    def monitor_batch(self, symbols: List[str]) -> List[Dict]:
        """批量監控股票"""
        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {
                executor.submit(self.get_signal, symbol): symbol for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                result = future.result()
                results.append(result)

        return results

    def display_dashboard(self):
        """顯示監控儀表板"""
        # 清屏
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 100)
        print("PPO UNIFIED STOCK MONITORING SYSTEM".center(100))
        print("=" * 100)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(100))
        print(
            f"Monitoring {len(self.symbols)} Stocks | Refresh: {self.refresh_interval}s".center(
                100
            )
        )
        print("=" * 100)

        # 分類顯示
        buy_signals = []
        sell_signals = []
        hold_signals = []
        error_signals = []

        for data in self.monitoring_data.values():
            if data["signal"] == "BUY":
                buy_signals.append(data)
            elif data["signal"] == "SELL":
                sell_signals.append(data)
            elif data["signal"] == "HOLD":
                hold_signals.append(data)
            else:
                error_signals.append(data)

        # 排序
        buy_signals.sort(key=lambda x: x["confidence"], reverse=True)
        sell_signals.sort(key=lambda x: x["confidence"], reverse=True)

        # 顯示強烈買入信號
        if buy_signals:
            print("\n" + "=" * 100)
            print("🟢 STRONG BUY SIGNALS".center(100))
            print("=" * 100)
            print(
                f"{'Symbol':<10} {'Price':<10} {'Change%':<10} {'Confidence':<12} {'Value':<10} {'Time':<10}"
            )
            print("-" * 100)
            for signal in buy_signals[:20]:  # 顯示前20個
                change_str = f"{signal.get('change', 0):+.2f}%"
                color = "\033[92m" if signal.get("change", 0) > 0 else "\033[91m"
                reset = "\033[0m"
                print(
                    f"{signal['symbol']:<10} "
                    f"${signal['price']:<9.2f} "
                    f"{color}{change_str:<10}{reset} "
                    f"{signal['confidence']*100:<11.1f}% "
                    f"{signal.get('value', 0):<10.2f} "
                    f"{signal.get('timestamp', 'N/A'):<10}"
                )

        # 顯示強烈賣出信號
        if sell_signals:
            print("\n" + "=" * 100)
            print("🔴 STRONG SELL SIGNALS".center(100))
            print("=" * 100)
            print(
                f"{'Symbol':<10} {'Price':<10} {'Change%':<10} {'Confidence':<12} {'Value':<10} {'Time':<10}"
            )
            print("-" * 100)
            for signal in sell_signals[:20]:  # 顯示前20個
                change_str = f"{signal.get('change', 0):+.2f}%"
                color = "\033[92m" if signal.get("change", 0) > 0 else "\033[91m"
                reset = "\033[0m"
                print(
                    f"{signal['symbol']:<10} "
                    f"${signal['price']:<9.2f} "
                    f"{color}{change_str:<10}{reset} "
                    f"{signal['confidence']*100:<11.1f}% "
                    f"{signal.get('value', 0):<10.2f} "
                    f"{signal.get('timestamp', 'N/A'):<10}"
                )

        # 統計摘要
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS".center(100))
        print("=" * 100)
        print(f"Total Monitored: {len(self.monitoring_data)}")
        print(
            f"Buy Signals: {len(buy_signals)} | Sell Signals: {len(sell_signals)} | Hold: {len(hold_signals)}"
        )
        if error_signals:
            print(f"Errors: {len(error_signals)}")

        # 顯示最高信心度的信號
        all_signals = buy_signals + sell_signals
        if all_signals:
            top_signal = max(all_signals, key=lambda x: x["confidence"])
            print(
                f"\nStrongest Signal: {top_signal['symbol']} - {top_signal['signal']} "
                f"(Confidence: {top_signal['confidence']*100:.1f}%)"
            )

        print("\n" + "=" * 100)
        print("Press Ctrl+C to stop monitoring".center(100))
        print("=" * 100)

    def run_monitoring(self):
        """運行監控循環"""
        self.running = True
        batch_size = 50  # 每批處理50個股票

        while self.running:
            try:
                print("\n[INFO] Fetching market data...")

                # 分批處理
                for i in range(0, len(self.symbols), batch_size):
                    batch = self.symbols[i : i + batch_size]
                    results = self.monitor_batch(batch)

                    # 更新監控數據
                    for result in results:
                        self.monitoring_data[result["symbol"]] = result

                    # 顯示進度
                    progress = min(i + batch_size, len(self.symbols))
                    print(
                        f"[PROGRESS] Processed {progress}/{len(self.symbols)} stocks..."
                    )

                # 顯示儀表板
                self.display_dashboard()

                # 等待下次刷新
                time.sleep(self.refresh_interval)

            except KeyboardInterrupt:
                print("\n[INFO] Stopping monitor...")
                self.running = False
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}")
                time.sleep(5)

    def start(self):
        """啟動監控系統"""
        print("\n" + "=" * 100)
        print("PPO UNIFIED MONITORING SYSTEM STARTING".center(100))
        print("=" * 100)

        # 載入模型和股票
        self.load_model()
        self.load_symbols()

        print(f"\n[INFO] Starting monitoring for {len(self.symbols)} stocks...")
        print(f"[INFO] Refresh interval: {self.refresh_interval} seconds")
        print("\n[INFO] Initializing market data...")

        # 開始監控
        self.run_monitoring()

        print("\n[INFO] Monitoring stopped")

        # 保存最後的信號
        self.save_signals()

    def save_signals(self):
        """保存交易信號"""
        if self.monitoring_data:
            filename = f"ppo_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(list(self.monitoring_data.values()), f, indent=2)
            print(f"[INFO] Signals saved to {filename}")


def main():
    """主程序"""
    monitor = UnifiedPPOMonitor()

    try:
        monitor.start()
    except Exception as e:
        print(f"\n[ERROR] System error: {str(e)}")
    finally:
        print("\n[INFO] System shutdown complete")


if __name__ == "__main__":
    main()
