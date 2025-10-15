#!/usr/bin/env python3
"""
Quick market data check via yfinance
- Fetches recent daily candles for a symbol (default AAPL)
- Prints last close; exits non-zero on failure
"""
import sys
import yfinance as yf


def main(symbol: str = "AAPL") -> int:
    try:
        yf.Ticker(symbol).history(period="5d", interval="1d")
        if data is None or data.empty:
            print(f"ERROR: empty data for {symbol}")
            return 2
        last_close = float(data["Close"].iloc[-1])
        print(f"OK: {symbol} last close = {last_close}")
        return 0
    except Exception as e:
        print(f"ERROR: failed to fetch {symbol}: {e}")
        return 1


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    raise SystemExit(main(sym))
