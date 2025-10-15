"""
Technical Indicators Performance Test
技術指標性能測試 - 測試 4000+ 股票的計算性能
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
import multiprocessing as mp
import psutil
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components
from src.indicators.indicator_calculator import IndicatorCalculator, CalculationConfig
from src.indicators.signal_generator import IndicatorSignalGenerator

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.metrics = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        return {
            "cpu_count": mp.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "python_version": sys.version.split()[0],
        }

    def profile_function(self, func, *args, **kwargs):
        """分析函數性能"""
        # 記錄開始狀態
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024**2)  # MB
        process.cpu_percent()
        start_time = time.time()

        # 執行函數
        result = func(*args, **kwargs)

        # 記錄結束狀態
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024**2)  # MB
        end_cpu = process.cpu_percent()

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        return result, {
            "execution_time": execution_time,
            "memory_used_mb": memory_used,
            "peak_memory_mb": end_memory,
            "cpu_percent": end_cpu,
        }


class DataGenerator:
    """測試數據生成器"""

    @staticmethod
    def generate_stock_universe(
        n_stocks: int = 4000, n_periods: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """
        生成股票池數據

        Args:
            n_stocks: 股票數量
            n_periods: 時間週期數（默認一年交易日）

        Returns:
            股票數據字典
        """
        print(f"Generating {n_stocks} stocks with {n_periods} periods each...")

        stocks_data = {}
        np.random.seed(42)  # 確保可重現性

        # 預定義一些股票特性
        stock_profiles = [
            {"volatility": 0.15, "trend": 0.08, "name": "growth"},  # 成長股
            {"volatility": 0.25, "trend": 0.12, "name": "tech"},  # 科技股
            {"volatility": 0.10, "trend": 0.05, "name": "utility"},  # 公用事業
            {"volatility": 0.20, "trend": 0.06, "name": "finance"},  # 金融股
            {"volatility": 0.30, "trend": 0.03, "name": "energy"},  # 能源股
            {"volatility": 0.18, "trend": 0.07, "name": "healthcare"},  # 醫療股
        ]

        for i in range(n_stocks):
            symbol = f"STOCK{i:04d}"

            # 選擇股票特性
            profile = stock_profiles[i % len(stock_profiles)]

            # 生成價格數據
            dates = pd.date_range(start="2023-01-01", periods=n_periods, freq="1D")

            # 使用幾何布朗運動模擬價格
            dt = 1 / 252  # 一天
            mu = profile["trend"]  # 年化收益率
            sigma = profile["volatility"]  # 年化波動率

            # 生成隨機遊走
            dW = np.random.normal(0, np.sqrt(dt), n_periods)

            # 初始價格
            S0 = np.random.uniform(10, 500)  # 10-500 美元

            # 幾何布朗運動
            returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
            prices = S0 * np.exp(np.cumsum(returns))

            # 生成 OHLC 數據
            high_noise = np.abs(np.random.normal(0, sigma * S0 * 0.01, n_periods))
            low_noise = np.abs(np.random.normal(0, sigma * S0 * 0.01, n_periods))

            pd.DataFrame(
                {
                    "open": prices + np.random.normal(0, sigma * S0 * 0.005, n_periods),
                    "high": prices + high_noise,
                    "low": prices - low_noise,
                    "close": prices,
                    "volume": np.random.lognormal(
                        mean=np.log(1000000), sigma=0.5, size=n_periods
                    ).astype(int),
                },
                index=dates,
            )

            # 確保 OHLC 邏輯正確
            data["high"] = np.maximum.reduce(
                [data["open"], data["high"], data["low"], data["close"]]
            )
            data["low"] = np.minimum.reduce(
                [data["open"], data["high"], data["low"], data["close"]]
            )

            stocks_data[symbol] = data

            # 進度提示
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{n_stocks} stocks...")

        print(f"✓ Generated {n_stocks} stocks successfully")
        return stocks_data


class PerformanceTestSuite:
    """性能測試套件"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.results = {
            "system_info": self.profiler.system_info,
            "test_results": {},
            "timestamp": datetime.now().isoformat(),
        }

    def test_single_stock_performance(self) -> Dict[str, Any]:
        """測試單股票指標計算性能"""
        print("\n🔍 Testing single stock indicator calculation performance...")

        # 生成單股測試數據
        test_data = DataGenerator.generate_stock_universe(n_stocks=1, n_periods=1000)
        stock_data = list(test_data.values())[0]

        # 初始化計算器
        config = CalculationConfig(
            timeframes=["1d"], use_multiprocessing=False, cache_results=False
        )
        calculator = IndicatorCalculator(config)

        # 測試各個指標的計算時間
        indicator_times = {}

        for indicator_name, indicator in calculator.indicators.items():
            try:
                result, metrics = self.profiler.profile_function(indicator.calculate, stock_data)

                indicator_times[indicator_name] = {
                    "execution_time_ms": metrics["execution_time"] * 1000,
                    "memory_used_mb": metrics["memory_used_mb"],
                    "result_size": len(result) if hasattr(result, "__len__") else 1,
                }

                print(f"  {indicator_name}: {metrics['execution_time']*1000:.2f}ms")

            except Exception as e:
                indicator_times[indicator_name] = {"error": str(e)}
                print(f"  {indicator_name}: ERROR - {e}")

        # 計算統計
        valid_times = [
            r["execution_time_ms"] for r in indicator_times.values() if "execution_time_ms" in r
        ]

        summary = {
            "total_indicators": len(calculator.indicators),
            "successful_calculations": len(valid_times),
            "avg_time_ms": np.mean(valid_times) if valid_times else 0,
            "max_time_ms": np.max(valid_times) if valid_times else 0,
            "min_time_ms": np.min(valid_times) if valid_times else 0,
            "total_time_ms": np.sum(valid_times) if valid_times else 0,
            "indicator_times": indicator_times,
        }

        print(f"✓ Single stock test completed - Avg: {summary['avg_time_ms']:.2f}ms per indicator")
        return summary

    def test_batch_calculation_performance(self) -> Dict[str, Any]:
        """測試批量計算性能"""
        print("\n📊 Testing batch calculation performance...")

        # 不同規模的測試
        test_scales = [10, 50, 100, 500, 1000]
        batch_results = {}

        for n_stocks in test_scales:
            print(f"  Testing {n_stocks} stocks...")

            # 生成測試數據
            stocks_data = DataGenerator.generate_stock_universe(
                n_stocks=n_stocks, n_periods=252  # 一年數據
            )

            # 配置計算器
            config = CalculationConfig(
                timeframes=["1d"],
                batch_size=min(50, n_stocks),
                use_multiprocessing=n_stocks > 50,
                max_workers=min(4, mp.cpu_count()),
                cache_results=True,
            )

            calculator = IndicatorCalculator(config)

            # 執行批量計算
            result, metrics = self.profiler.profile_function(
                calculator.calculate_all_indicators, stocks_data
            )

            # 計算性能指標
            batch_results[n_stocks] = {
                "execution_time_s": metrics["execution_time"],
                "memory_used_mb": metrics["memory_used_mb"],
                "peak_memory_mb": metrics["peak_memory_mb"],
                "stocks_per_second": n_stocks / metrics["execution_time"],
                "indicators_per_second": (n_stocks * len(calculator.indicators))
                / metrics["execution_time"],
                "memory_per_stock_mb": metrics["peak_memory_mb"] / n_stocks,
                "successful_calculations": len([r for r in result.values() if r]),
            }

            print(
                f"    {n_stocks} stocks: {metrics['execution_time']:.2f}s, "
                f"{batch_results[n_stocks]['stocks_per_second']:.1f} stocks/s"
            )

        return batch_results

    def test_multiprocessing_scaling(self) -> Dict[str, Any]:
        """測試多進程擴展性能"""
        print("\n⚡ Testing multiprocessing scaling performance...")

        # 生成固定測試數據
        test_stocks = 200
        stocks_data = DataGenerator.generate_stock_universe(n_stocks=test_stocks, n_periods=252)

        scaling_results = {}
        worker_counts = [1, 2, 4, mp.cpu_count()]

        for workers in worker_counts:
            print(f"  Testing with {workers} worker(s)...")

            config = CalculationConfig(
                timeframes=["1d"],
                batch_size=25,
                use_multiprocessing=workers > 1,
                max_workers=workers,
                cache_results=False,  # 避免緩存影響
            )

            calculator = IndicatorCalculator(config)

            # 執行計算
            result, metrics = self.profiler.profile_function(
                calculator.calculate_all_indicators, stocks_data
            )

            scaling_results[workers] = {
                "execution_time_s": metrics["execution_time"],
                "speedup": None,  # 稍後計算
                "efficiency": None,  # 稍後計算
                "memory_used_mb": metrics["memory_used_mb"],
                "successful_calculations": len([r for r in result.values() if r]),
            }

            print(f"    {workers} workers: {metrics['execution_time']:.2f}s")

        # 計算加速比和效率
        baseline_time = scaling_results[1]["execution_time_s"]
        for workers in scaling_results:
            time_taken = scaling_results[workers]["execution_time_s"]
            speedup = baseline_time / time_taken
            efficiency = speedup / workers

            scaling_results[workers]["speedup"] = speedup
            scaling_results[workers]["efficiency"] = efficiency

        return scaling_results

    def test_memory_efficiency(self) -> Dict[str, Any]:
        """測試記憶體效率"""
        print("\n💾 Testing memory efficiency...")

        memory_results = {}
        stock_counts = [100, 500, 1000, 2000]

        for n_stocks in stock_counts:
            print(f"  Testing memory usage with {n_stocks} stocks...")

            # 監控記憶體使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)  # MB

            # 生成數據
            stocks_data = DataGenerator.generate_stock_universe(n_stocks=n_stocks, n_periods=252)

            data_memory = process.memory_info().rss / (1024**2) - initial_memory

            # 計算指標
            config = CalculationConfig(
                timeframes=["1d"], use_multiprocessing=False, cache_results=False
            )
            calculator = IndicatorCalculator(config)

            results = calculator.calculate_all_indicators(stocks_data)

            final_memory = process.memory_info().rss / (1024**2)
            calculation_memory = final_memory - initial_memory - data_memory

            memory_results[n_stocks] = {
                "data_memory_mb": data_memory,
                "calculation_memory_mb": calculation_memory,
                "total_memory_mb": final_memory - initial_memory,
                "memory_per_stock_mb": (final_memory - initial_memory) / n_stocks,
                "memory_efficiency": calculation_memory / data_memory if data_memory > 0 else 0,
            }

            print(
                f"    {n_stocks} stocks: Data: {data_memory:.1f}MB, "
                f"Calc: {calculation_memory:.1f}MB, Total: {final_memory - initial_memory:.1f}MB"
            )

            # 清理記憶體
            del stocks_data
            del results

        return memory_results

    def test_large_scale_performance(self) -> Dict[str, Any]:
        """測試大規模性能（4000+ 股票）"""
        print("\n🚀 Testing large-scale performance (4000+ stocks)...")

        # 生成 4000 股票數據
        n_stocks = 4000
        print(f"Generating {n_stocks} stocks (this may take a few minutes)...")

        stocks_data = DataGenerator.generate_stock_universe(n_stocks=n_stocks, n_periods=252)

        # 配置計算器以獲得最佳性能
        config = CalculationConfig(
            timeframes=["1d"],
            batch_size=100,
            use_multiprocessing=True,
            max_workers=min(8, mp.cpu_count()),
            cache_results=True,
        )

        calculator = IndicatorCalculator(config)

        print(f"Calculating indicators for {n_stocks} stocks...")

        # 執行大規模計算
        result, metrics = self.profiler.profile_function(
            calculator.calculate_all_indicators, stocks_data
        )

        # 獲取性能統計
        perf_stats = calculator.get_performance_stats()

        large_scale_results = {
            "total_stocks": n_stocks,
            "execution_time_s": metrics["execution_time"],
            "memory_used_mb": metrics["memory_used_mb"],
            "peak_memory_mb": metrics["peak_memory_mb"],
            "stocks_per_second": n_stocks / metrics["execution_time"],
            "total_indicators_calculated": perf_stats["total_calculations"],
            "cache_hit_rate": perf_stats["cache_hit_rate"],
            "avg_time_per_stock_ms": (metrics["execution_time"] * 1000) / n_stocks,
            "memory_per_stock_mb": metrics["peak_memory_mb"] / n_stocks,
            "successful_stocks": len([r for r in result.values() if r]),
            "success_rate": len([r for r in result.values() if r]) / n_stocks,
        }

        print("✓ Large-scale test completed:")
        print(f"  Time: {metrics['execution_time']:.1f}s")
        print(f"  Speed: {large_scale_results['stocks_per_second']:.1f} stocks/s")
        print(f"  Memory: {metrics['peak_memory_mb']:.1f}MB peak")
        print(f"  Success rate: {large_scale_results['success_rate']*100:.1f}%")

        return large_scale_results

    def run_all_tests(self) -> Dict[str, Any]:
        """運行所有性能測試"""
        print("Technical Indicators Performance Test Suite")
        print("=" * 60)
        print(
            f"🖥️  System: {self.profiler.system_info['cpu_count']} CPUs, "
            f"{self.profiler.system_info['memory_total_gb']}GB RAM"
        )
        print("-" * 60)

        try:
            # 運行各項測試
            self.results["test_results"]["single_stock"] = self.test_single_stock_performance()
            self.results["test_results"][
                "batch_calculation"
            ] = self.test_batch_calculation_performance()
            self.results["test_results"][
                "multiprocessing_scaling"
            ] = self.test_multiprocessing_scaling()
            self.results["test_results"]["memory_efficiency"] = self.test_memory_efficiency()
            self.results["test_results"]["large_scale"] = self.test_large_scale_performance()

            # 生成性能報告
            self.generate_performance_report()

            print("\n🎉 All performance tests completed successfully!")

        except Exception as e:
            print(f"\n❌ Performance test failed: {e}")
            logger.error(f"Performance test error: {e}")
            self.results["error"] = str(e)

        return self.results

    def generate_performance_report(self):
        """生成性能報告"""
        report_file = Path(__file__).parent / "performance_test_results.json"

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # 生成簡化報告
        summary_file = Path(__file__).parent / "performance_summary.txt"

        with open(summary_file, "w") as f:
            f.write("Technical Indicators Performance Test Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Test Date: {self.results['timestamp']}\n")
            f.write(
                f"System: {self.profiler.system_info['cpu_count']} CPUs, "
                f"{self.profiler.system_info['memory_total_gb']}GB RAM\n\n"
            )

            # 大規模測試結果
            if "large_scale" in self.results["test_results"]:
                large = self.results["test_results"]["large_scale"]
                f.write("🚀 Large-Scale Performance (4000 stocks):\n")
                f.write(f"  Execution Time: {large['execution_time_s']:.1f} seconds\n")
                f.write(f"  Processing Speed: {large['stocks_per_second']:.1f} stocks/second\n")
                f.write(f"  Memory Usage: {large['peak_memory_mb']:.1f} MB\n")
                f.write(f"  Success Rate: {large['success_rate']*100:.1f}%\n\n")

            # 單股票性能
            if "single_stock" in self.results["test_results"]:
                single = self.results["test_results"]["single_stock"]
                f.write("🔍 Single Stock Performance:\n")
                f.write(f"  Average per indicator: {single['avg_time_ms']:.2f}ms\n")
                f.write(f"  Total indicators: {single['total_indicators']}\n")
                f.write(
                    f"  Success rate: {single['successful_calculations']}/{single['total_indicators']}\n\n"
                )

            # 多進程擴展性
            if "multiprocessing_scaling" in self.results["test_results"]:
                scaling = self.results["test_results"]["multiprocessing_scaling"]
                f.write("⚡ Multiprocessing Scaling:\n")
                for workers, data in scaling.items():
                    f.write(
                        f"  {workers} workers: {data['speedup']:.2f}x speedup, "
                        f"{data['efficiency']*100:.1f}% efficiency\n"
                    )

        print(f"\n📊 Performance report saved to: {report_file}")
        print(f"📋 Summary saved to: {summary_file}")


def main():
    """主函數"""
    test_suite = PerformanceTestSuite()
    results = test_suite.run_all_tests()

    # 檢查是否達到性能要求
    requirements_met = True

    if "large_scale" in results["test_results"]:
        large_scale = results["test_results"]["large_scale"]

        # 性能要求檢查
        if large_scale["stocks_per_second"] < 50:  # 至少50股票/秒
            print("⚠️  Warning: Processing speed below target (50 stocks/second)")
            requirements_met = False

        if large_scale["success_rate"] < 0.95:  # 至少95%成功率
            print("⚠️  Warning: Success rate below target (95%)")
            requirements_met = False

        if large_scale["peak_memory_mb"] > 8000:  # 不超過8GB記憶體
            print("⚠️  Warning: Memory usage above target (8GB)")
            requirements_met = False

    if requirements_met:
        print("\n✅ All performance requirements met!")
        return True
    else:
        print("\n❌ Some performance requirements not met.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
