"""
Test Free Data Sources - Plan B
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_alpaca():
    """Test Alpaca Markets connection"""
    print("\n" + "=" * 60)
    print("Testing Alpaca Markets Connection")
    print("=" * 60)

    try:
        from alpaca_trade_api import REST

        api_key = os.getenv("ALPACA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            print("[ERROR] Missing Alpaca credentials")
            return False

        base_url = "https://paper-api.alpaca.markets"
        api = REST(api_key, secret_key, base_url, api_version="v2")

        account = api.get_account()
        print(f"[OK] Connected to Alpaca Markets")
        print(f"     Account Status: {account.status}")
        print(f"     Buying Power: ${float(account.buying_power):,.2f}")

        # Test real-time data
        bars = api.get_bars("AAPL", "1Min", limit=5).df
        if not bars.empty:
            print(f"[OK] Real-time data working")
            print(f"     Latest AAPL: ${bars['close'].iloc[-1]:.2f}")

        return True

    except ImportError:
        print("[ERROR] alpaca-trade-api not installed")
        print("        Run: pip install alpaca-trade-api")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return False


def test_yahoo():
    """Test Yahoo Finance"""
    print("\n" + "=" * 60)
    print("Testing Yahoo Finance")
    print("=" * 60)

    try:
        import yfinance as yf

        ticker = yf.Ticker("AAPL")
        info = ticker.info

        print(f"[OK] Yahoo Finance working")
        print(f"     AAPL Price: ${info.get('currentPrice', 0):.2f}")

        hist = ticker.history(period="5d")
        if not hist.empty:
            print(f"[OK] Historical data available")
            print(f"     5-day records: {len(hist)}")

        return True

    except ImportError:
        print("[ERROR] yfinance not installed")
        print("        Run: pip install yfinance")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return False


def test_alpha_vantage():
    """Test Alpha Vantage Free API"""
    print("\n" + "=" * 60)
    print("Testing Alpha Vantage Free API")
    print("=" * 60)

    try:
        import requests

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            print("[ERROR] Missing Alpha Vantage API key")
            return False

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "AAPL",
            "outputsize": "compact",
            "apikey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "Time Series (Daily)" in data:
            print(f"[OK] Alpha Vantage connected")
            print(f"     Free tier: 5 calls/min, 500/day")
            return True
        elif "Note" in data:
            print(f"[WARNING] Rate limit reached")
            return True
        else:
            print(f"[ERROR] API response issue")
            return False

    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return False


def test_capital():
    """Test Capital.com connection"""
    print("\n" + "=" * 60)
    print("Testing Capital.com Connection")
    print("=" * 60)

    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.connectors.capital_com_api import CapitalComAPI

        api = CapitalComAPI()
        if api.authenticate():
            print(f"[OK] Capital.com connected")

            accounts = api.get_accounts()
            if accounts:
                acc = accounts[0] if isinstance(accounts, list) else accounts
                balance = acc.get("balance", {})
                print(f"     Balance: ${balance.get('balance', 0):,.2f}")

            return True
        else:
            print(f"[ERROR] Authentication failed")
            return False

    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return False


def main():
    print("\n" + "=" * 80)
    print("FREE DATA SOURCES TEST - PLAN B (ZERO COST)")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test each source
    results = {
        "Alpaca Markets": test_alpaca(),
        "Yahoo Finance": test_yahoo(),
        "Alpha Vantage": test_alpha_vantage(),
        "Capital.com": test_capital(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for source, status in results.items():
        status_text = "[OK]" if status else "[FAIL]"
        print(f"{status_text} {source}")

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\nSuccess Rate: {success_rate:.0f}%")

    if success_rate == 100:
        print("\n[SUCCESS] All data sources working!")
        print("\nNext Steps:")
        print("1. Build data pipeline")
        print("2. Start data collection")
        print("3. Begin strategy testing")
    elif success_rate >= 75:
        print("\n[WARNING] Some issues detected")
    else:
        print("\n[ERROR] Multiple failures")

    # Save results
    with open("test_results.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "plan": "B",
                "results": results,
                "success_rate": success_rate,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
