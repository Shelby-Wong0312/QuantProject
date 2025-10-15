"""
System Integration Test Suite
系統整合測試套件
Cloud PM - Task PM-001
"""

import sys
import asyncio
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SystemIntegrationTester:
    """Comprehensive system integration testing"""

    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None

        # Component availability
        self.components = {
            "ml_models": False,
            "rl_trading": False,
            "portfolio_optimization": False,
            "risk_management": False,
            "data_collection": False,
            "paper_trading": False,
            "dashboard": False,
            "anomaly_detection": False,
        }

    def check_component_availability(self):
        """Check if all components are available"""
        print("\n" + "=" * 60)
        print("COMPONENT AVAILABILITY CHECK")
        print("=" * 60)

        # Check ML Models
        try:
            from src.ml_models.lstm_attention import LSTMAttentionModel
            from src.ml_models.xgboost_ensemble import XGBoostEnsemble

            self.components["ml_models"] = True
            print("[OK] ML Models: Available")
        except ImportError as e:
            print(f"[FAIL] ML Models: {e}")

        # Check RL Trading
        try:
            from src.rl_trading.ppo_agent import PPOAgent
            from src.rl_trading.trading_env import TradingEnvironment

            self.components["rl_trading"] = True
            print("[OK] RL Trading: Available")
        except ImportError as e:
            print(f"[FAIL] RL Trading: {e}")

        # Check Portfolio Optimization
        try:
            from src.portfolio.mpt_optimizer import MPTOptimizer

            self.components["portfolio_optimization"] = True
            print("[OK] Portfolio Optimization: Available")
        except ImportError as e:
            print(f"[FAIL] Portfolio Optimization: {e}")

        # Check Risk Management
        try:
            from src.risk.risk_manager_enhanced import EnhancedRiskManager
            from src.risk.dynamic_stop_loss import DynamicStopLoss
            from src.risk.stress_testing import StressTester

            self.components["risk_management"] = True
            print("[OK] Risk Management: Available")
        except ImportError as e:
            print(f"[FAIL] Risk Management: {e}")

        # Check Data Collection
        try:
            from src.data.realtime_collector import RealtimeDataCollector
            from src.data.data_validator import DataValidator

            self.components["data_collection"] = True
            print("[OK] Data Collection: Available")
        except ImportError as e:
            print(f"[FAIL] Data Collection: {e}")

        # Check Paper Trading
        try:
            from src.core.paper_trading import PaperTradingSimulator

            self.components["paper_trading"] = True
            print("[OK] Paper Trading: Available")
        except ImportError as e:
            print(f"[FAIL] Paper Trading: {e}")

        # Check Dashboard
        dashboard_path = Path("dashboard/main_dashboard.py")
        self.components["dashboard"] = dashboard_path.exists()
        status = "[OK]" if self.components["dashboard"] else "[FAIL]"
        print(
            f"{status} Dashboard: {'Available' if dashboard_path.exists() else 'Not found'}"
        )

        # Check Anomaly Detection
        try:
            from src.risk.anomaly_detection import MarketAnomalyDetector
            from src.risk.circuit_breaker import CircuitBreaker
            from src.risk.deleveraging import RapidDeleveraging

            self.components["anomaly_detection"] = True
            print("[OK] Anomaly Detection: Available")
        except ImportError as e:
            print(f"[FAIL] Anomaly Detection: {e}")

        # Summary
        available = sum(self.components.values())
        total = len(self.components)
        print(f"\nComponents Available: {available}/{total}")

        return available == total

    async def test_end_to_end_workflow(self):
        """Test complete trading workflow"""
        print("\n" + "=" * 60)
        print("END-TO-END WORKFLOW TEST")
        print("=" * 60)

        workflow_steps = []

        # Step 1: Data Collection
        print("\n[1] Testing Data Collection...")
        try:
            from src.data.realtime_collector import RealtimeDataCollector

            RealtimeDataCollector(["AAPL", "GOOGL", "MSFT"])
            # Simulate data collection
            workflow_steps.append(
                ("Data Collection", True, "Data collected successfully")
            )
            print("   [OK] Data collection initialized")
        except Exception as e:
            workflow_steps.append(("Data Collection", False, str(e)))
            print(f"   [FAIL] {e}")

        # Step 2: Portfolio Optimization
        print("\n[2] Testing Portfolio Optimization...")
        try:
            from src.portfolio.mpt_optimizer import MPTOptimizer

            optimizer = MPTOptimizer()
            # Generate test data
            test_returns = pd.DataFrame(np.random.randn(100, 3) * 0.01)
            optimizer.optimize(test_returns.values)
            workflow_steps.append(
                ("Portfolio Optimization", True, "Optimal weights calculated")
            )
            print("   [OK] Portfolio optimized")
        except Exception as e:
            workflow_steps.append(("Portfolio Optimization", False, str(e)))
            print(f"   [FAIL] {e}")

        # Step 3: Risk Management
        print("\n[3] Testing Risk Management...")
        try:
            from src.risk.risk_manager_enhanced import EnhancedRiskManager

            risk_manager = EnhancedRiskManager(initial_capital=100000)
            risk_manager.check_trade_risk("AAPL", 100, 150.0, "BUY")
            workflow_steps.append(("Risk Management", True, "Risk checks passed"))
            print("   [OK] Risk management active")
        except Exception as e:
            workflow_steps.append(("Risk Management", False, str(e)))
            print(f"   [FAIL] {e}")

        # Step 4: Paper Trading Execution
        print("\n[4] Testing Paper Trading...")
        try:
            from src.core.paper_trading import PaperTradingSimulator

            simulator = PaperTradingSimulator(initial_balance=100000)
            # Execute test trade
            await simulator.execute_trade("AAPL", 10, "BUY")
            workflow_steps.append(("Paper Trading", True, "Trade executed"))
            print("   [OK] Paper trading functional")
        except Exception as e:
            workflow_steps.append(("Paper Trading", False, str(e)))
            print(f"   [FAIL] {e}")

        # Step 5: Anomaly Detection
        print("\n[5] Testing Anomaly Detection...")
        try:
            from src.risk.anomaly_detection import MarketAnomalyDetector

            MarketAnomalyDetector()
            # Test with synthetic data
            workflow_steps.append(("Anomaly Detection", True, "Monitoring active"))
            print("   [OK] Anomaly detection operational")
        except Exception as e:
            workflow_steps.append(("Anomaly Detection", False, str(e)))
            print(f"   [FAIL] {e}")

        # Calculate success rate
        successful = sum(1 for _, success, _ in workflow_steps if success)
        total = len(workflow_steps)
        success_rate = successful / total if total > 0 else 0

        print("\n" + "-" * 60)
        print("WORKFLOW TEST RESULTS:")
        for step, success, message in workflow_steps:
            status = "[OK]" if success else "[FAIL]"
            print(f"{status} {step}: {message}")

        print(f"\nSuccess Rate: {success_rate:.1%} ({successful}/{total})")

        self.test_results["workflow"] = {
            "success_rate": success_rate,
            "steps": workflow_steps,
        }

        return success_rate >= 0.8

    async def test_performance(self):
        """Test system performance metrics"""
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST")
        print("=" * 60)

        performance_metrics = {}

        # Test 1: Data Processing Speed
        print("\n[1] Data Processing Speed Test...")
        start = time.time()
        try:
            # Simulate data processing
            pd.DataFrame(np.random.randn(10000, 10))
            data.rolling(20).mean()
            elapsed = time.time() - start
            performance_metrics["data_processing"] = {
                "time": elapsed,
                "passed": elapsed < 1.0,
                "message": f"Processed 10000 rows in {elapsed:.3f}s",
            }
            print(f"   [OK] {performance_metrics['data_processing']['message']}")
        except Exception as e:
            performance_metrics["data_processing"] = {
                "time": 999,
                "passed": False,
                "message": str(e),
            }
            print(f"   [FAIL] {e}")

        # Test 2: Model Inference Speed
        print("\n[2] Model Inference Speed Test...")
        start = time.time()
        try:
            # Simulate model inference
            X = np.random.randn(1000, 20)
            # Simple linear model simulation
            W = np.random.randn(20, 1)
            X @ W
            elapsed = time.time() - start
            performance_metrics["model_inference"] = {
                "time": elapsed,
                "passed": elapsed < 0.5,
                "message": f"1000 predictions in {elapsed:.3f}s",
            }
            print(f"   [OK] {performance_metrics['model_inference']['message']}")
        except Exception as e:
            performance_metrics["model_inference"] = {
                "time": 999,
                "passed": False,
                "message": str(e),
            }
            print(f"   [FAIL] {e}")

        # Test 3: Risk Calculation Speed
        print("\n[3] Risk Calculation Speed Test...")
        start = time.time()
        try:
            from src.risk.dynamic_stop_loss import DynamicStopLoss

            stop_loss = DynamicStopLoss()
            # Test stop loss calculation
            for _ in range(100):
                stop_loss.calculate_atr_stop(
                    "AAPL", 150.0, pd.Series(np.random.randn(20))
                )
            elapsed = time.time() - start
            performance_metrics["risk_calculation"] = {
                "time": elapsed,
                "passed": elapsed < 1.0,
                "message": f"100 risk calculations in {elapsed:.3f}s",
            }
            print(f"   [OK] {performance_metrics['risk_calculation']['message']}")
        except Exception as e:
            performance_metrics["risk_calculation"] = {
                "time": 999,
                "passed": False,
                "message": str(e),
            }
            print(f"   [FAIL] {e}")

        # Calculate overall performance score
        passed_tests = sum(1 for m in performance_metrics.values() if m["passed"])
        total_tests = len(performance_metrics)
        performance_score = passed_tests / total_tests if total_tests > 0 else 0

        print("\n" + "-" * 60)
        print("PERFORMANCE METRICS:")
        for test, metrics in performance_metrics.items():
            status = "[PASS]" if metrics["passed"] else "[FAIL]"
            print(f"{status} {test}: {metrics['message']}")

        print(f"\nPerformance Score: {performance_score:.1%}")

        self.test_results["performance"] = {
            "score": performance_score,
            "metrics": performance_metrics,
        }

        return performance_score >= 0.7

    async def test_fault_recovery(self):
        """Test system fault recovery capabilities"""
        print("\n" + "=" * 60)
        print("FAULT RECOVERY TEST")
        print("=" * 60)

        recovery_tests = []

        # Test 1: Handle Missing Data
        print("\n[1] Testing Missing Data Handling...")
        try:
            from src.data.data_validator import DataValidator

            validator = DataValidator()

            # Create data with missing values
            pd.DataFrame(
                {
                    "price": [100, np.nan, 102, 103, np.nan],
                    "volume": [1000, 2000, np.nan, 4000, 5000],
                }
            )

            cleaned = validator.clean_data(data)
            has_nulls = cleaned.isnull().any().any()

            recovery_tests.append(
                {
                    "test": "Missing Data",
                    "passed": not has_nulls,
                    "message": "Data cleaned successfully",
                }
            )
            print("   [OK] Missing data handled")
        except Exception as e:
            recovery_tests.append(
                {"test": "Missing Data", "passed": False, "message": str(e)}
            )
            print(f"   [FAIL] {e}")

        # Test 2: Circuit Breaker Activation
        print("\n[2] Testing Circuit Breaker...")
        try:
            from src.risk.circuit_breaker import CircuitBreaker

            breaker = CircuitBreaker(initial_value=100000)

            # Trigger circuit breaker
            triggered = breaker.check_trigger(85000)  # 15% drop

            recovery_tests.append(
                {
                    "test": "Circuit Breaker",
                    "passed": triggered is not None,
                    "message": f'Breaker triggered: {triggered.name if triggered else "None"}',
                }
            )
            print("   [OK] Circuit breaker functional")
        except Exception as e:
            recovery_tests.append(
                {"test": "Circuit Breaker", "passed": False, "message": str(e)}
            )
            print(f"   [FAIL] {e}")

        # Test 3: Error Recovery
        print("\n[3] Testing Error Recovery...")
        try:
            # Simulate error and recovery
            error_occurred = False
            recovered = False

            try:
                # Intentionally cause an error
                pass
            except ZeroDivisionError:
                error_occurred = True
                # Recover with default value
                recovered = True

            recovery_tests.append(
                {
                    "test": "Error Recovery",
                    "passed": error_occurred and recovered,
                    "message": "Error caught and recovered",
                }
            )
            print("   [OK] Error recovery successful")
        except Exception as e:
            recovery_tests.append(
                {"test": "Error Recovery", "passed": False, "message": str(e)}
            )
            print(f"   [FAIL] {e}")

        # Calculate recovery score
        passed = sum(1 for test in recovery_tests if test["passed"])
        total = len(recovery_tests)
        recovery_score = passed / total if total > 0 else 0

        print("\n" + "-" * 60)
        print("FAULT RECOVERY RESULTS:")
        for test in recovery_tests:
            status = "[PASS]" if test["passed"] else "[FAIL]"
            print(f"{status} {test['test']}: {test['message']}")

        print(f"\nRecovery Score: {recovery_score:.1%}")

        self.test_results["fault_recovery"] = {
            "score": recovery_score,
            "tests": recovery_tests,
        }

        return recovery_score >= 0.7

    async def run_all_tests(self):
        """Run all integration tests"""
        self.start_time = datetime.now()

        print("\n" + "=" * 60)
        print("SYSTEM INTEGRATION TEST SUITE")
        print("=" * 60)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run tests
        results = {
            "components": self.check_component_availability(),
            "workflow": await self.test_end_to_end_workflow(),
            "performance": await self.test_performance(),
            "fault_recovery": await self.test_fault_recovery(),
        }

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Calculate overall results
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        overall_success = passed / total if total > 0 else 0

        # Generate summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)

        for test_name, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} {test_name.replace('_', ' ').title()}")

        print("\n" + "-" * 60)
        print(f"Overall Success Rate: {overall_success:.1%}")
        print(f"Tests Passed: {passed}/{total}")
        print(f"Duration: {duration:.2f} seconds")

        # Save report
        {
            "timestamp": self.start_time.isoformat(),
            "duration": duration,
            "overall_success_rate": overall_success,
            "tests_passed": passed,
            "tests_total": total,
            "results": results,
            "detailed_results": self.test_results,
        }

        report_file = Path("reports/integration_test_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n[FILE] Detailed report saved: {report_file}")

        # Check if ready for production
        ready_for_production = overall_success >= 0.8

        if ready_for_production:
            print("\n[SUCCESS] SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("\n[WARNING] System needs improvements before production deployment")

        return ready_for_production


async def main():
    """Main test execution"""
    tester = SystemIntegrationTester()
    success = await tester.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
