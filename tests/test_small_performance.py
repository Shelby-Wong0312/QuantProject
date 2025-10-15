"""
Small-scale Performance Test
小規模性能測試 - 避免運行大型測試
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components
from src.indicators.indicator_calculator import IndicatorCalculator, CalculationConfig


def generate_test_data(n_stocks: int = 50, n_periods: int = 100) -> Dict[str, pd.DataFrame]:
    """生成小規模測試數據"""
    stocks_data = {}
    np.random.seed(42)

    for i in range(n_stocks):
        symbol = f"TEST{i:03d}"
        dates = pd.date_range(start="2024-01-01", periods=n_periods, freq="1D")

        # 生成隨機價格數據
        price = 100 + np.cumsum(np.random.randn(n_periods) * 0.02)

        stocks_data[symbol] = pd.DataFrame(
            {
                "open": price + np.random.randn(n_periods) * 0.01,
                "high": price + np.abs(np.random.randn(n_periods) * 0.02),
                "low": price - np.abs(np.random.randn(n_periods) * 0.02),
                "close": price,
                "volume": np.random.randint(10000, 100000, n_periods),
            },
            index=dates,
        )

        # 確保 OHLC 邏輯正確
        stocks_data[symbol]["high"] = np.maximum.reduce(
            [
                stocks_data[symbol]["open"],
                stocks_data[symbol]["high"],
                stocks_data[symbol]["low"],
                stocks_data[symbol]["close"],
            ]
        )
        stocks_data[symbol]["low"] = np.minimum.reduce(
            [
                stocks_data[symbol]["open"],
                stocks_data[symbol]["high"],
                stocks_data[symbol]["low"],
                stocks_data[symbol]["close"],
            ]
        )

    return stocks_data


def test_small_scale_performance():
    """測試小規模性能"""
    print("Small-Scale Technical Indicators Performance Test")
    print("=" * 55)

    # 測試不同規模
    test_cases = [
        {"stocks": 10, "periods": 100, "name": "Small"},
        {"stocks": 25, "periods": 100, "name": "Medium"},
        {"stocks": 50, "periods": 100, "name": "Large"},
    ]

    results = {}

    for case in test_cases:
        print(f"\nTesting {case['name']} scale: {case['stocks']} stocks, {case['periods']} periods")

        # 生成測試數據
        start_time = time.time()
        stocks_data = generate_test_data(case["stocks"], case["periods"])
        data_gen_time = time.time() - start_time

        # 配置計算器
        config = CalculationConfig(
            timeframes=["1d"],
            batch_size=min(10, case["stocks"]),
            use_multiprocessing=case["stocks"] > 20,
            cache_results=True,
        )

        calculator = IndicatorCalculator(config)

        # 執行計算
        calc_start = time.time()
        indicator_results = calculator.calculate_all_indicators(stocks_data)
        calc_time = time.time() - calc_start

        # 生成信號
        signal_start = time.time()
        calculator.calculate_signals(indicator_results)
        signal_time = time.time() - signal_start

        # 統計結果
        successful_stocks = len([r for r in indicator_results.values() if r])
        total_indicators = len(calculator.indicators)

        case_result = {
            "stocks": case["stocks"],
            "periods": case["periods"],
            "data_generation_time": data_gen_time,
            "calculation_time": calc_time,
            "signal_generation_time": signal_time,
            "total_time": data_gen_time + calc_time + signal_time,
            "successful_stocks": successful_stocks,
            "success_rate": successful_stocks / case["stocks"],
            "stocks_per_second": case["stocks"] / calc_time if calc_time > 0 else 0,
            "indicators_per_second": (
                (case["stocks"] * total_indicators) / calc_time if calc_time > 0 else 0
            ),
        }

        results[case["name"]] = case_result

        print(f"  Data generation: {data_gen_time:.3f}s")
        print(f"  Indicator calculation: {calc_time:.3f}s")
        print(f"  Signal generation: {signal_time:.3f}s")
        print(f"  Success rate: {case_result['success_rate']*100:.1f}%")
        print(f"  Processing speed: {case_result['stocks_per_second']:.1f} stocks/second")

    # 測試單個指標性能
    print("\nTesting individual indicator performance...")
    single_stock_data = generate_test_data(1, 252)  # 一年數據
    stock_data = list(single_stock_data.values())[0]

    config = CalculationConfig(timeframes=["1d"], use_multiprocessing=False)
    calculator = IndicatorCalculator(config)

    indicator_times = {}
    for name, indicator in calculator.indicators.items():
        try:
            start_time = time.time()
            result = indicator.calculate(stock_data)
            end_time = time.time()

            indicator_times[name] = {
                "time_ms": (end_time - start_time) * 1000,
                "result_size": len(result) if hasattr(result, "__len__") else 1,
            }

        except Exception as e:
            indicator_times[name] = {"error": str(e)}

    # 顯示最快和最慢的指標
    valid_times = {k: v["time_ms"] for k, v in indicator_times.items() if "time_ms" in v}
    if valid_times:
        fastest = min(valid_times, key=valid_times.get)
        slowest = max(valid_times, key=valid_times.get)

        print(f"  Fastest indicator: {fastest} ({valid_times[fastest]:.2f}ms)")
        print(f"  Slowest indicator: {slowest} ({valid_times[slowest]:.2f}ms)")
        print(f"  Average time per indicator: {np.mean(list(valid_times.values())):.2f}ms")

    # 保存結果
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "scale_tests": results,
        "individual_indicators": indicator_times,
        "summary": {
            "total_indicators_tested": len(calculator.indicators),
            "avg_processing_speed": np.mean([r["stocks_per_second"] for r in results.values()]),
            "best_success_rate": max([r["success_rate"] for r in results.values()]),
        },
    }

    results_file = Path(__file__).parent / "small_performance_results.json"
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    print("\nTest Summary:")
    print(f"  Total indicators tested: {len(calculator.indicators)}")
    print(
        f"  Average processing speed: {test_results['summary']['avg_processing_speed']:.1f} stocks/second"
    )
    print(f"  Best success rate: {test_results['summary']['best_success_rate']*100:.1f}%")
    print(f"\nResults saved to: {results_file}")

    # 檢查性能要求
    min_speed_required = 5  # 至少5股票/秒
    min_success_rate = 0.9  # 至少90%成功率

    speed_ok = test_results["summary"]["avg_processing_speed"] >= min_speed_required
    success_ok = test_results["summary"]["best_success_rate"] >= min_success_rate

    if speed_ok and success_ok:
        print("\nSUCCESS: Performance requirements met!")
        return True
    else:
        if not speed_ok:
            print(f"\nWARNING: Processing speed below requirement ({min_speed_required} stocks/s)")
        if not success_ok:
            print(f"\nWARNING: Success rate below requirement ({min_success_rate*100}%)")
        return False


if __name__ == "__main__":
    success = test_small_scale_performance()
    exit(0 if success else 1)
