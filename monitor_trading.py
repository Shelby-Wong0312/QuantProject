"""
Monitor Live Trading System
"""

import time
from src.connectors.capital_com_api import CapitalComAPI
import sqlite3


def monitor():
    print("\n" + "=" * 60)
    print("LIVE TRADING MONITOR")
    print("=" * 60)

    # Check API
    api = CapitalComAPI()
    if api.authenticate():
        print("[OK] API Connected")
        account = api.get_account_info()
        if account:
            print(f"[OK] Balance: ${account.get('balance', 0):,.2f}")
            print(f"[OK] Equity: ${account.get('equity', 0):,.2f}")
    else:
        print("[FAIL] API Connection Failed")

    # Check recent trades
    try:
        conn = sqlite3.connect("data/live_trades.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM trades WHERE date(timestamp) = date('now')"
        )
        today_trades = cursor.fetchone()[0]
        print(f"\n[TRADES] Today's trades: {today_trades}")

        # Get last 5 trades
        cursor.execute(
            "SELECT timestamp, symbol, action, quantity, price, pnl FROM trades ORDER BY id DESC LIMIT 5"
        )
        trades = cursor.fetchall()
        if trades:
            print("\nRecent Trades:")
            for trade in trades:
                timestamp, symbol, action, qty, price, pnl = trade
                pnl_str = f"P&L: ${pnl:.2f}" if pnl else ""
                print(
                    f"  {timestamp[:16]} | {action:4} {qty:3} {symbol:5} @ ${price:.2f} {pnl_str}"
                )
        conn.close()
    except Exception:
        print("[INFO] No trade database yet")

    # Check market prices
    print("\nCurrent Prices:")
    ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    for symbol in symbols:
        price = api.get_market_price(symbol)
        if price:
            print(f"  {symbol}: ${price:.2f}")

    print("\n[SYSTEM] Trading system is running...")
    print("Press Ctrl+C to exit monitor")


if __name__ == "__main__":
    while True:
        monitor()
        time.sleep(30)  # Update every 30 seconds
