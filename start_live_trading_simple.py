"""
Simplified Live Trading System Starter
修復策略初始化問題的簡化版本
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


async def main():
    """Simplified main entry point"""
    print("=" * 60)
    print("STARTING SIMPLIFIED LIVE TRADING SYSTEM")
    print("=" * 60)

    try:
        # Import the live trading system
        from src.live_trading.live_system import LiveTradingSystem

        # Initialize system
        system = LiveTradingSystem()
        print("[OK] Trading system initialized")

        # For simplified version, use mock strategies
        class SimpleMomentumStrategy:
            """Simplified momentum strategy without config requirements"""

            def __init__(self):
                self.name = "SimpleMomentum"

            def generate_signals(self, data):
                """Generate simple signals"""
                if data is None or data.empty:
                    return 0
                # Simple logic: buy if price went up
                if len(data) > 2:
                    return 1 if data["Close"].iloc[-1] > data["Close"].iloc[-2] else -1
                return 0

            def calculate_position_size(self, signal, capital):
                """Simple position sizing"""
                return min(100, int(capital * 0.1 / 100))  # 10% of capital

            def risk_management(self, position):
                """Simple risk management"""
                return {"stop_loss": 0.02, "take_profit": 0.05}

        class SimpleMeanReversionStrategy:
            """Simplified mean reversion strategy without config requirements"""

            def __init__(self):
                self.name = "SimpleMeanReversion"

            def generate_signals(self, data):
                """Generate simple signals"""
                if data is None or data.empty or len(data) < 20:
                    return 0
                # Simple logic: buy if price below 20-day average
                sma = data["Close"].rolling(window=20).mean()
                if data["Close"].iloc[-1] < sma.iloc[-1] * 0.98:
                    return 1
                elif data["Close"].iloc[-1] > sma.iloc[-1] * 1.02:
                    return -1
                return 0

            def calculate_position_size(self, signal, capital):
                """Simple position sizing"""
                return min(100, int(capital * 0.1 / 100))  # 10% of capital

            def risk_management(self, position):
                """Simple risk management"""
                return {"stop_loss": 0.02, "take_profit": 0.03}

        # Create simple strategies
        strategies = [SimpleMomentumStrategy(), SimpleMeanReversionStrategy()]
        print(f"[OK] Loaded {len(strategies)} strategies")

        # Define symbols to trade
        ["AAPL", "MSFT", "GOOGL"]
        print(f"[OK] Monitoring {len(symbols)} symbols: {', '.join(symbols)}")

        # Start trading with smaller capital for testing
        capital = 10000
        print(f"[OK] Starting with capital: ${capital:,.2f}")

        print("\n" + "=" * 60)
        print("LIVE TRADING ACTIVE")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Start trading
        await system.start_trading(strategies, symbols, capital=capital)

    except KeyboardInterrupt:
        print("\n[!] Trading interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Trading system error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n" + "=" * 60)
        print("TRADING SYSTEM STOPPED")
        print("=" * 60)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        sys.exit(1)

    # Check for required packages
    try:
        import pandas
        import numpy
        import yfinance
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Please run: pip install pandas numpy yfinance python-dotenv")
        sys.exit(1)

    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
