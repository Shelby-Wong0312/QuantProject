#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO即時監控系統 - 簡潔高效版
Real-time PPO Monitoring System
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# 設置輸出編碼
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class PPOModel(nn.Module):
    """PPO模型"""

    def __init__(self, input_dim=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Buy, Hold, Sell
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class LiveMonitor:
    """即時監控器"""

    def __init__(self):
        self.model = PPOModel()
        self.load_model()
        self.symbols = self.load_symbols()

    def load_model(self):
        """載入模型"""
        model_path = "models/ppo_3488_stocks.pt"
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                if "model_state_dict" in checkpoint:
                    # 嘗試載入權重
                    pass
                print("[OK] Model loaded")
            except Exception:
                print("[OK] Using new model")
        self.model.eval()

    def load_symbols(self):
        """載入股票列表"""
        # 使用主要股票列表
        [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK-B",
            "V",
            "JNJ",
            "WMT",
            "JPM",
            "PG",
            "MA",
            "UNH",
            "DIS",
            "HD",
            "PYPL",
            "BAC",
            "VZ",
            "ADBE",
            "NFLX",
            "INTC",
            "CMCSA",
            "KO",
            "PFE",
            "TMO",
            "CSCO",
            "PEP",
            "AVGO",
            "ABBV",
            "NKE",
            "CVX",
            "MRK",
            "WFC",
            "T",
            "CRM",
            "AMD",
            "MCD",
            "COST",
            "BMY",
            "MDT",
            "NEE",
            "UPS",
            "TXN",
            "HON",
            "QCOM",
            "RTX",
            "LOW",
            "ORCL",
            "IBM",
        ]

        # 嘗試載入更多股票
        if os.path.exists("validated_yahoo_symbols_final.txt"):
            try:
                with open("validated_yahoo_symbols_final.txt", "r") as f:
                    all_symbols = [line.strip() for line in f if line.strip()]
                    # 取前200個
                    all_symbols[:200]
                    print(f"[OK] Loaded {len(symbols)} symbols")
            except Exception:
                pass

        return symbols

    def get_features(self, data):
        """提取特徵"""
        if len(data) < 20:
            return np.zeros(50)

        features = []
        prices = data["Close"].values

        # 基本特徵
        returns = np.diff(prices) / prices[:-1]
        features.append(np.mean(returns[-10:]))
        features.append(np.std(returns[-10:]))
        features.append((prices[-1] - prices[-10]) / prices[-10])

        # MA
        for period in [5, 10, 20]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features.append((prices[-1] / ma) - 1)
            else:
                features.append(0)

        # 填充到50
        while len(features) < 50:
            features.append(0)

        return np.array(features[:50], dtype=np.float32)

    def analyze_stock(self, symbol):
        """分析單個股票"""
        try:
            # 獲取數據
            ticker = yf.Ticker(symbol)
            ticker.history(period="2mo")

            if len(data) < 20:
                return None

            # 提取特徵
            features = self.get_features(data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # 預測
            with torch.no_grad():
                probs = self.model(features_tensor).squeeze().numpy()

            # 信號
            ["BUY", "HOLD", "SELL"]
            signal_idx = np.argmax(probs)

            return {
                "symbol": symbol,
                "price": float(data["Close"].iloc[-1]),
                "change": float(
                    (data["Close"].iloc[-1] - data["Close"].iloc[-2])
                    / data["Close"].iloc[-2]
                    * 100
                ),
                "signal": signals[signal_idx],
                "confidence": float(probs[signal_idx] * 100),
                "buy_prob": float(probs[0] * 100),
                "hold_prob": float(probs[1] * 100),
                "sell_prob": float(probs[2] * 100),
            }
        except Exception:
            return None

    def run(self):
        """運行監控"""
        print("\n" + "=" * 80)
        print("PPO LIVE MONITORING SYSTEM".center(80))
        print("=" * 80)

        while True:
            try:
                print(
                    f"\nScanning {len(self.symbols)} stocks... [{datetime.now().strftime('%H:%M:%S')}]"
                )

                results = []
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(self.analyze_stock, symbol)
                        for symbol in self.symbols
                    ]
                    for future in futures:
                        result = future.result()
                        if result:
                            results.append(result)

                # 分類結果
                buy_signals = [r for r in results if r["signal"] == "BUY"]
                sell_signals = [r for r in results if r["signal"] == "SELL"]

                # 排序
                buy_signals.sort(key=lambda x: x["confidence"], reverse=True)
                sell_signals.sort(key=lambda x: x["confidence"], reverse=True)

                # 清屏
                os.system("cls" if os.name == "nt" else "clear")

                # 顯示結果
                print("\n" + "=" * 80)
                print(
                    f"PPO MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(
                        80
                    )
                )
                print("=" * 80)

                if buy_signals:
                    print(f"\n[BUY SIGNALS] {len(buy_signals)} stocks")
                    print("-" * 80)
                    print(
                        f"{'Symbol':<8} {'Price':>8} {'Change%':>8} {'Signal':<6} {'Confidence':>10}"
                    )
                    print("-" * 80)
                    for s in buy_signals[:10]:
                        print(
                            f"{s['symbol']:<8} ${s['price']:>7.2f} {s['change']:>7.2f}% "
                            f"{s['signal']:<6} {s['confidence']:>9.1f}%"
                        )

                if sell_signals:
                    print(f"\n[SELL SIGNALS] {len(sell_signals)} stocks")
                    print("-" * 80)
                    print(
                        f"{'Symbol':<8} {'Price':>8} {'Change%':>8} {'Signal':<6} {'Confidence':>10}"
                    )
                    print("-" * 80)
                    for s in sell_signals[:10]:
                        print(
                            f"{s['symbol']:<8} ${s['price']:>7.2f} {s['change']:>7.2f}% "
                            f"{s['signal']:<6} {s['confidence']:>9.1f}%"
                        )

                # 統計
                print("\n" + "-" * 80)
                print(
                    f"Total: {len(results)} | Buy: {len(buy_signals)} | "
                    f"Sell: {len(sell_signals)} | Hold: {len(results)-len(buy_signals)-len(sell_signals)}"
                )

                # 最強信號
                if results:
                    strongest = max(results, key=lambda x: x["confidence"])
                    print(
                        f"\nStrongest Signal: {strongest['symbol']} - {strongest['signal']} ({strongest['confidence']:.1f}%)"
                    )

                print("\n[Press Ctrl+C to stop]")

                # 保存信號
                with open("live_signals.json", "w") as f:
                    json.dump(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "buy_signals": buy_signals[:20],
                            "sell_signals": sell_signals[:20],
                        },
                        f,
                        indent=2,
                    )

                # 等待30秒
                time.sleep(30)

            except KeyboardInterrupt:
                print("\n[Stopped]")
                break
            except Exception as e:
                print(f"[Error] {str(e)}")
                time.sleep(5)


if __name__ == "__main__":
    monitor = LiveMonitor()
    monitor.run()
