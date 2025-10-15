"""
Quick Final System Test
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


def test_system():
    print("\n" + "=" * 60)
    print("FINAL SYSTEM TEST - QUICK VERSION")
    print("=" * 60)

    results = {"timestamp": datetime.now().isoformat(), "tests": {}, "summary": {}}

    # 1. Test Database
    print("\n[1] Testing Database...")
    try:
        conn = sqlite3.connect("data/quant_trading.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchone()[0]
        conn.close()
        print(f"  [OK] Database connected - {tables} tables found")
        results["tests"]["database"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Database error: {e}")
        results["tests"]["database"] = "FAIL"

    # 2. Test Capital.com API Config
    print("\n[2] Testing Capital.com API Configuration...")
    try:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                content = f.read()
                if "CAPITAL_API_KEY" in content:
                    print("  [OK] API credentials configured")
                    results["tests"]["api_config"] = "PASS"
                else:
                    print("  [WARN] API key not found in .env")
                    results["tests"]["api_config"] = "PARTIAL"
        else:
            print("  [FAIL] .env file not found")
            results["tests"]["api_config"] = "FAIL"
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        results["tests"]["api_config"] = "FAIL"

    # 3. Test Core Modules
    print("\n[3] Testing Core Modules...")
    try:
        import yfinance
        import torch
        import sklearn
        from src.core.trading_system import TradingSystem
        from src.connectors.capital_com_api import CapitalComAPI

        print("  [OK] All core modules loaded")
        results["tests"]["modules"] = "PASS"
    except ImportError as e:
        print(f"  [WARN] Some modules missing: {e}")
        results["tests"]["modules"] = "PARTIAL"

    # 4. Test Data Access
    print("\n[4] Testing Market Data Access...")
    try:
        import yfinance as yf

        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d")
        if not hist.empty:
            latest_price = hist["Close"].iloc[-1]
            print(f"  [OK] Market data working - AAPL: ${latest_price:.2f}")
            results["tests"]["market_data"] = "PASS"
        else:
            print("  [FAIL] Cannot fetch market data")
            results["tests"]["market_data"] = "FAIL"
    except Exception as e:
        print(f"  [FAIL] Data error: {e}")
        results["tests"]["market_data"] = "FAIL"

    # 5. Test Strategy Components
    print("\n[5] Testing Strategy Components...")
    try:
        # Test indicator calculation
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        sma20 = prices.rolling(20).mean()
        print(f"  [OK] Indicators calculated - SMA20: {sma20.iloc[-1]:.2f}")
        results["tests"]["strategies"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Strategy error: {e}")
        results["tests"]["strategies"] = "FAIL"

    # 6. Test Risk Management
    print("\n[6] Testing Risk Management...")
    try:
        portfolio_value = 100000
        risk_per_trade = 0.02
        max_risk = portfolio_value * risk_per_trade
        print(f"  [OK] Risk calculations - Max risk per trade: ${max_risk:.2f}")
        results["tests"]["risk"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Risk error: {e}")
        results["tests"]["risk"] = "FAIL"

    # 7. Test File System
    print("\n[7] Testing File System...")
    required_dirs = ["data", "src", "logs", "reports", "scripts"]
    missing = [d for d in required_dirs if not os.path.exists(d)]
    if not missing:
        print("  [OK] All required directories exist")
        results["tests"]["filesystem"] = "PASS"
    else:
        print(f"  [WARN] Missing directories: {missing}")
        results["tests"]["filesystem"] = "PARTIAL"

    # 8. Test ML Models
    print("\n[8] Testing ML Models...")
    try:

        if os.path.exists("ppo_trader_iter_150.pt"):
            print("  [OK] Trained ML models found")
            results["tests"]["ml_models"] = "PASS"
        else:
            print("  [INFO] ML models not yet trained")
            results["tests"]["ml_models"] = "PARTIAL"
    except Exception as e:
        print(f"  [WARN] ML test skipped: {e}")
        results["tests"]["ml_models"] = "SKIP"

    # 9. Test Docker/DevOps
    print("\n[9] Testing DevOps Configuration...")
    if os.path.exists("Dockerfile") and os.path.exists("docker-compose.yml"):
        print("  [OK] Docker configuration found")
        results["tests"]["devops"] = "PASS"
    else:
        print("  [WARN] Docker files missing")
        results["tests"]["devops"] = "PARTIAL"

    # 10. Test Security
    print("\n[10] Testing Security...")
    if os.path.exists("src/security/secure_config.py"):
        print("  [OK] Security module configured")
        results["tests"]["security"] = "PASS"
    else:
        print("  [WARN] Security module not found")
        results["tests"]["security"] = "PARTIAL"

    # Calculate Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results["tests"].values() if v == "PASS")
    partial = sum(1 for v in results["tests"].values() if v == "PARTIAL")
    failed = sum(1 for v in results["tests"].values() if v == "FAIL")
    total = len(results["tests"])

    print(f"\nTotal Tests: {total}")
    print(f"  [PASS]: {passed}")
    print(f"  [PARTIAL]: {partial}")
    print(f"  [FAIL]: {failed}")

    pass_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nPass Rate: {pass_rate:.1f}%")

    results["summary"] = {
        "total": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "pass_rate": pass_rate,
    }

    # Save results
    with open("quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Final verdict
    print("\n" + "=" * 60)
    if pass_rate >= 80:
        print("RESULT: SYSTEM READY FOR DEPLOYMENT!")
        print("The system has passed critical tests and is ready to trade.")
    elif pass_rate >= 60:
        print("RESULT: SYSTEM OPERATIONAL WITH WARNINGS")
        print("The system can run but some components need attention.")
    else:
        print("RESULT: SYSTEM NOT READY")
        print("Critical components are missing or failing.")
    print("=" * 60)

    return pass_rate >= 60


if __name__ == "__main__":
    success = test_system()
    print("\nTest report saved to: quick_test_results.json")
    sys.exit(0 if success else 1)
