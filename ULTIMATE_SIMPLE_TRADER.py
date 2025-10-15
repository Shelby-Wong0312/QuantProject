#!/usr/bin/env python3
"""
ULTIMATE SIMPLE AUTO-TRADER
一個檔案搞定4000+股票監控和自動交易
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# 載入環境變量
load_dotenv()


class SimpleAutoTrader:
    def __init__(self):
        # Capital.com API設定
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER")
        self.password = os.getenv("CAPITAL_API_PASSWORD")
        self.base_url = "https://demo-api-capital.backend-capital.com"
        self.session_token = None
        self.cst = None

        # 交易參數
        self.min_rsi = 30  # RSI超賣
        self.max_rsi = 70  # RSI超買
        self.volume_spike = 2.0  # 成交量突增倍數
        self.position_size = 0.01  # 每筆交易1%資金
        self.stop_loss = 0.02  # 2%止損
        self.take_profit = 0.05  # 5%止盈

        # 載入股票列表
        self.stocks = self.load_stocks()
        self.positions = {}
        print(f"[READY] Monitoring {len(self.stocks)} stocks")

    def load_stocks(self):
        """Load stock list"""
        # Try to load from file
        if os.path.exists("data/all_symbols.txt"):
            with open("data/all_symbols.txt", "r") as f:
                stocks = [line.strip() for line in f.readlines() if line.strip()]
                return stocks[:4000]  # Max 4000 stocks

        # Default stock list
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "AMD",
            "NFLX",
            "BABA",
            "JPM",
            "BAC",
            "WMT",
            "DIS",
            "V",
            "MA",
            "PYPL",
            "SQ",
            "COIN",
            "ROKU",
            "SNAP",
            "UBER",
            "LYFT",
            "ZM",
            "DOCU",
        ]

    def login_capital(self):
        """Login to Capital.com"""
        headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"identifier": self.identifier, "password": self.password}

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/session", headers=headers, json=payload, timeout=10
            )
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.session_token = response.headers.get("X-SECURITY-TOKEN")
                print("[OK] Capital.com connected")
                return True
        except:
            pass

        print("[INFO] Capital.com not connected, using simulation")
        return False

    def get_signals(self, symbol):
        """Get trading signals"""
        try:
            # 獲取數據
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")

            if len(hist) < 14:
                return None

            # 計算指標
            # RSI
            delta = hist["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # 成交量
            avg_volume = hist["Volume"].mean()
            current_volume = hist["Volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume

            # 價格
            current_price = hist["Close"].iloc[-1]
            ma20 = hist["Close"].rolling(20).mean().iloc[-1]

            # 生成信號
            signal = None
            strength = 0

            # 買入信號
            if current_rsi < self.min_rsi and volume_ratio > self.volume_spike:
                signal = "BUY"
                strength = (self.min_rsi - current_rsi) / self.min_rsi
            # 賣出信號
            elif current_rsi > self.max_rsi:
                signal = "SELL"
                strength = (current_rsi - self.max_rsi) / (100 - self.max_rsi)
            # 價格突破
            elif current_price > ma20 * 1.02 and volume_ratio > 1.5:
                signal = "BUY"
                strength = 0.5

            if signal:
                return {
                    "symbol": symbol,
                    "signal": signal,
                    "strength": strength,
                    "price": current_price,
                    "rsi": current_rsi,
                    "volume_ratio": volume_ratio,
                }

        except Exception as e:
            pass

        return None

    def execute_trade(self, signal):
        """Execute trade"""
        symbol = signal["symbol"]

        # Check if position exists
        if symbol in self.positions:
            if signal["signal"] == "SELL":
                print(f"[CLOSE] Close position {symbol} @ ${signal['price']:.2f}")
                self.positions.pop(symbol)
            return

        # Open new position
        if signal["signal"] == "BUY" and len(self.positions) < 20:
            self.positions[symbol] = {
                "entry_price": signal["price"],
                "stop_loss": signal["price"] * (1 - self.stop_loss),
                "take_profit": signal["price"] * (1 + self.take_profit),
                "time": datetime.now(),
            }
            print(
                f"[BUY] {symbol} @ ${signal['price']:.2f} (RSI:{signal['rsi']:.1f}, Vol:{signal['volume_ratio']:.1f}x)"
            )

            # If Capital.com connected, execute real trade
            if self.cst:
                self.place_capital_order(symbol, "BUY", 1)

    def place_capital_order(self, symbol, direction, size):
        """Place order on Capital.com"""
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.session_token,
            "Content-Type": "application/json",
        }

        payload = {"epic": f"US.{symbol}.CASH", "direction": direction, "size": size}

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/positions", headers=headers, json=payload, timeout=10
            )
            if response.status_code == 200:
                print(f"[CAPITAL.COM] Order success: {symbol}")
        except:
            pass

    def check_positions(self):
        """Check stop loss and take profit for positions"""
        for symbol in list(self.positions.keys()):
            try:
                current = yf.Ticker(symbol).info.get("regularMarketPrice", 0)
                position = self.positions[symbol]

                # 止損
                if current <= position["stop_loss"]:
                    print(f"[STOP LOSS] {symbol} @ ${current:.2f}")
                    self.positions.pop(symbol)
                # 止盈
                elif current >= position["take_profit"]:
                    print(f"[TAKE PROFIT] {symbol} @ ${current:.2f}")
                    self.positions.pop(symbol)

            except:
                pass

    def scan_market(self):
        """Scan market for signals"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(self.stocks)} stocks...")

        signals = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.get_signals, symbol) for symbol in self.stocks]
            for future in futures:
                result = future.result()
                if result:
                    signals.append(result)

        # Sort by signal strength
        signals.sort(key=lambda x: x["strength"], reverse=True)

        # Execute top 5 strongest signals
        for signal in signals[:5]:
            self.execute_trade(signal)

        if signals:
            print(f"[SIGNALS] Found {len(signals)} signals")

        # Check existing positions
        self.check_positions()

        if self.positions:
            print(
                f"[POSITIONS] Holding {len(self.positions)} positions: {list(self.positions.keys())}"
            )

    def run(self):
        """Main program"""
        print("\n" + "=" * 60)
        print(" ULTIMATE SIMPLE AUTO-TRADER")
        print(" Monitor 4000+ Stocks | Auto Signals | Auto Trade")
        print("=" * 60)

        # Connect to Capital.com
        self.login_capital()

        # Main loop
        while True:
            try:
                self.scan_market()
                time.sleep(30)  # Scan every 30 seconds
            except KeyboardInterrupt:
                print("\n[EXIT] System stopped")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)


if __name__ == "__main__":
    trader = SimpleAutoTrader()
    trader.run()
