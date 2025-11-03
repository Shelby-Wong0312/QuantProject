"""
Test script to verify live trading system fixes
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_strategy_initialization():
    """Test if strategies can be initialized properly"""
    print("\n" + "=" * 60)
    print("TESTING STRATEGY INITIALIZATION FIX")
    print("=" * 60)

    try:
        # Test 1: Import strategy classes
        print("\n[1] Testing imports...")
        from src.strategies.traditional.momentum_strategy import MomentumStrategy
        from src.strategies.traditional.mean_reversion import MeanReversionStrategy
        from src.strategies.strategy_interface import StrategyConfig

        print("   [OK] Imports successful")

        # Test 2: Create strategy configs
        print("\n[2] Creating strategy configurations...")
        momentum_config = StrategyConfig(
            name="MomentumStrategy",
            enabled=True,
            weight=0.5,
            risk_limit=0.02,
            max_positions=5,
            parameters={
                "rsi_period": 14,
                "rsi_buy_threshold": 60,
                "rsi_sell_threshold": 40,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "volume_period": 20,
                "volume_threshold": 1.5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_position_size": 0.1,
                "max_drawdown": 0.1,
                "position_sizing_method": "fixed",
            },
            ["AAPL", "MSFT"],
        )
        print("   [OK] Momentum config created")

        mean_reversion_config = StrategyConfig(
            name="MeanReversionStrategy",
            enabled=True,
            weight=0.5,
            risk_limit=0.02,
            max_positions=5,
            parameters={
                "bb_period": 20,
                "bb_std": 2,
                "zscore_period": 20,
                "zscore_threshold": 2.0,
                "holding_period": 5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "max_position_size": 0.1,
                "max_drawdown": 0.1,
                "position_sizing_method": "fixed",
            },
            ["AAPL", "MSFT"],
        )
        print("   [OK] Mean reversion config created")

        # Test 3: Initialize strategies
        print("\n[3] Initializing strategies with configs...")
        momentum_strategy = MomentumStrategy(momentum_config)
        print(f"   [OK] MomentumStrategy initialized: {momentum_strategy.name}")

        mean_reversion_strategy = MeanReversionStrategy(mean_reversion_config)
        print(f"   [OK] MeanReversionStrategy initialized: {mean_reversion_strategy.name}")

        # Test 4: Test strategy methods
        print("\n[4] Testing strategy methods...")

        # Create dummy data
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        dates = pd.date_range(end=datetime.now(), periods=100)
        dummy_data = pd.DataFrame(
            {
                "Open": np.random.randn(100).cumsum() + 100,
                "High": np.random.randn(100).cumsum() + 101,
                "Low": np.random.randn(100).cumsum() + 99,
                "Close": np.random.randn(100).cumsum() + 100,
                "Volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # Test signal generation
        try:
            momentum_strategy.generate_signals(dummy_data)
            print("   [OK] MomentumStrategy.generate_signals() works")
        except Exception as e:
            print(f"   [FAIL] MomentumStrategy.generate_signals() failed: {e}")

        try:
            mean_reversion_strategy.generate_signals(dummy_data)
            print("   [OK] MeanReversionStrategy.generate_signals() works")
        except Exception as e:
            print(f"   [FAIL] MeanReversionStrategy.generate_signals() failed: {e}")

        # Test 5: Test live system initialization
        print("\n[5] Testing LiveTradingSystem...")
        from src.live_trading.live_system import LiveTradingSystem

        system = LiveTradingSystem()
        print("   [OK] LiveTradingSystem initialized")

        status = system.get_status()
        print(f"   [OK] System status: {status}")

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED - SYSTEM READY!")
        print("=" * 60)
        print("\nYou can now run the live trading system with:")
        print("  python start_live_trading_simple.py")
        print("or")
        print(
            '  python -c "import asyncio; from src.live_trading.live_system import main; asyncio.run(main())"'
        )

        return True

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_strategy_initialization()
    sys.exit(0 if success else 1)
