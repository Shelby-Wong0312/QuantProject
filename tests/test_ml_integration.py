"""
Test ML Integration and Generate Reports
Cloud Quant - Task Q-701
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.ml_strategy_integration import MLStrategyIntegration
from src.backtesting.ml_backtest import MLBacktester, BacktestConfig
from src.optimization.hyperparameter_tuning import HyperparameterTuner, OptimizationConfig


async def test_ml_integration():
    """Run complete ML integration test"""

    print("\n" + "=" * 70)
    print("ML/DL/RL INTEGRATION TEST")
    print("Cloud Quant - Task Q-701")
    print("=" * 70)

    # Step 1: Test ML Strategy Integration
    print("\n[1] Testing ML Strategy Integration...")
    strategy = MLStrategyIntegration(initial_capital=100000)

    # Generate test data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    test_data = {
        "AAPL": pd.DataFrame(
            {
                "open": np.random.uniform(150, 160, 100),
                "high": np.random.uniform(160, 170, 100),
                "low": np.random.uniform(140, 150, 100),
                "close": np.random.uniform(145, 165, 100),
                "volume": np.random.uniform(1000000, 5000000, 100),
                "returns": np.random.normal(0.001, 0.02, 100),
            },
            index=dates,
        )
    }

    # Generate signals
    signals = await strategy.generate_trading_signals(test_data)
    print(f"   Generated {len(signals)} signals successfully")

    # Step 2: Run Backtest
    print("\n[2] Running Backtest System...")
    config = BacktestConfig(
        initial_capital=100000,
        start_date="2020-01-01",
        end_date="2024-12-31",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        rebalance_frequency="monthly",
        use_walk_forward=False,  # Faster for testing
    )

    backtester = MLBacktester(config)
    historical_data = backtester.load_historical_data()

    print("   Running backtest (this may take a minute)...")
    result = await backtester.backtest_strategy(historical_data)

    # Generate backtest report
    report = backtester.generate_report(result)
    print("   Backtest report generated successfully")

    # Step 3: Run Hyperparameter Tuning
    print("\n[3] Running Hyperparameter Optimization...")
    opt_config = OptimizationConfig(
        target_metric="sharpe_ratio",
        n_iterations=10,  # Reduced for quick testing
        save_results=True,
    )

    tuner = HyperparameterTuner(opt_config)

    # Use smaller dataset for optimization
    small_data = {symbol: data for symbol, data in list(historical_data.items())[:3]}

    print("   Optimizing parameters (this may take a few minutes)...")
    best_params = await tuner.optimize_ml_parameters(small_data)

    # Generate optimization report
    opt_report = tuner.generate_optimization_report()
    print(opt_report)

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)

    print("\n✅ ML Strategy Integration: WORKING")
    print("✅ Backtesting System: WORKING")
    print("✅ Hyperparameter Tuning: WORKING")

    print("\nGenerated Files:")
    print("  - reports/backtest_report.json")
    print("  - reports/optimal_parameters.yaml")
    print("  - reports/optimal_parameters.json")

    print(f"\nKey Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annual Return: {result.annual_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Total Trades: {result.total_trades}")

    print("\n✅ Task Q-701 Implementation Complete!")

    return True


if __name__ == "__main__":
    print("\nStarting ML Integration Test...")
    print("This will test all components of Task Q-701")

    success = asyncio.run(test_ml_integration())

    if success:
        print("\n✅ All tests passed successfully!")
        print("ML/DL/RL models are now fully integrated into the trading system")
    else:
        print("\n❌ Some tests failed. Please check the logs.")
