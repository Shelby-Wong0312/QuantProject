"""
Simple Automated Trading System
簡化版自動交易系統 - 確保能正常運行
"""

import time
import random
import sqlite3
import os
from datetime import datetime
import yfinance as yf


class SimpleTradingSystem:
    def __init__(self):
        self.positions = {}
        self.cash = 140370.87
        self.max_positions = 10
        self.position_size = 0.05  # 5% per trade

        # Top 50 most traded stocks
        self.symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "V",
            "JNJ",
            "WMT",
            "JPM",
            "PG",
            "MA",
            "UNH",
            "DIS",
            "HD",
            "BAC",
            "XOM",
            "PFE",
            "ABBV",
            "KO",
            "PEP",
            "TMO",
            "COST",
            "AVGO",
            "CSCO",
            "NKE",
            "MRK",
            "CVX",
            "ABT",
            "LLY",
            "ADBE",
            "CRM",
            "ORCL",
            "ACN",
            "VZ",
            "NFLX",
            "INTC",
            "AMD",
            "T",
            "WFC",
            "MDT",
            "UPS",
            "MS",
            "BMY",
            "RTX",
            "QCOM",
            "NEE",
            "HON",
        ]

        self.init_database()
        print(f"[INIT] System initialized with ${self.cash:,.2f}")
        print(f"[INIT] Monitoring {len(self.symbols)} stocks")

    def init_database(self):
        """Initialize database"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        conn = sqlite3.connect("data/simple_trades.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                quantity INTEGER,
                price REAL,
                total REAL
            )
        """
        )
        conn.commit()
        conn.close()

    def get_price(self, symbol):
        """Get current price"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except:
            pass
        return None

    def analyze_stock(self, symbol):
        """Simple analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if len(hist) < 2:
                return "HOLD"

            # Simple momentum strategy
            current = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            change_pct = (current - prev) / prev

            # Random component for demo
            rand = random.random()

            if change_pct > 0.02 and rand < 0.1:  # 2% up + 10% random chance
                return "BUY"
            elif change_pct < -0.02 and symbol in self.positions and rand < 0.2:
                return "SELL"

            # Check positions for exit
            if symbol in self.positions:
                entry_price = self.positions[symbol]["price"]
                pnl_pct = (current - entry_price) / entry_price

                if pnl_pct <= -0.05:  # Stop loss at -5%
                    return "SELL"
                elif pnl_pct >= 0.10:  # Take profit at 10%
                    return "SELL"

            return "HOLD"

        except:
            return "HOLD"

    def execute_trade(self, symbol, action, price):
        """Execute trade"""
        if action == "BUY" and len(self.positions) < self.max_positions:
            # Calculate shares
            position_value = self.cash * self.position_size
            shares = int(position_value / price)

            if shares > 0 and shares * price <= self.cash:
                cost = shares * price
                self.cash -= cost
                self.positions[symbol] = {"shares": shares, "price": price, "time": datetime.now()}

                # Save to database
                self.save_trade(symbol, "BUY", shares, price)

                print(f"[BUY] {shares} shares of {symbol} at ${price:.2f} (Total: ${cost:.2f})")
                return True

        elif action == "SELL" and symbol in self.positions:
            position = self.positions[symbol]
            shares = position["shares"]
            revenue = shares * price
            self.cash += revenue

            pnl = revenue - (shares * position["price"])
            pnl_pct = pnl / (shares * position["price"]) * 100

            del self.positions[symbol]

            # Save to database
            self.save_trade(symbol, "SELL", shares, price)

            print(
                f"[SELL] {shares} shares of {symbol} at ${price:.2f} (P&L: ${pnl:.2f}, {pnl_pct:.1f}%)"
            )
            return True

        return False

    def save_trade(self, symbol, action, quantity, price):
        """Save trade to database"""
        conn = sqlite3.connect("data/simple_trades.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (timestamp, symbol, action, quantity, price, total)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (datetime.now().isoformat(), symbol, action, quantity, price, quantity * price),
        )
        conn.commit()
        conn.close()

    def display_status(self):
        """Display current status"""
        portfolio_value = self.cash
        for symbol, pos in self.positions.items():
            current_price = self.get_price(symbol)
            if current_price:
                portfolio_value += pos["shares"] * current_price

        print("\n" + "=" * 60)
        print(f"STATUS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Positions: {len(self.positions)}/{self.max_positions}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")

        if self.positions:
            print("\nOpen Positions:")
            for symbol, pos in self.positions.items():
                current_price = self.get_price(symbol)
                if current_price:
                    pnl = (current_price - pos["price"]) * pos["shares"]
                    pnl_pct = (current_price - pos["price"]) / pos["price"] * 100
                    print(
                        f"  {symbol}: {pos['shares']} shares @ ${pos['price']:.2f} | Current: ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.1f}%)"
                    )

        print("=" * 60)

    def run(self):
        """Main loop"""
        print("\n[START] Trading system is running...")
        print("[INFO] Press Ctrl+C to stop\n")

        cycle = 0
        while True:
            try:
                cycle += 1
                print(f"\n[SCAN] Cycle {cycle} - Scanning {len(self.symbols)} stocks...")

                trades_executed = 0
                signals_generated = 0

                # Scan stocks
                for symbol in self.symbols:
                    price = self.get_price(symbol)
                    if not price:
                        continue

                    signal = self.analyze_stock(symbol)

                    if signal != "HOLD":
                        signals_generated += 1
                        if self.execute_trade(symbol, signal, price):
                            trades_executed += 1

                print(
                    f"[SCAN] Complete - {signals_generated} signals, {trades_executed} trades executed"
                )

                # Display status every 5 cycles
                if cycle % 5 == 0:
                    self.display_status()

                # Wait before next scan
                print(f"[WAIT] Next scan in 60 seconds...")
                time.sleep(60)

            except KeyboardInterrupt:
                print("\n[STOP] Shutting down...")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        # Final status
        self.display_status()
        print("\n[END] Trading system stopped")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("     SIMPLE AUTOMATED TRADING SYSTEM")
    print("=" * 60)
    print("\nThis system will:")
    print("1. Monitor 50 most traded stocks")
    print("2. Generate buy/sell signals")
    print("3. Execute trades automatically")
    print("4. Apply 5% stop loss and 10% take profit")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    system = SimpleTradingSystem()
    system.run()
