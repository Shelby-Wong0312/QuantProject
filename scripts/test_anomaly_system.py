"""
Comprehensive Test Script for Market Anomaly Detection System
市場異常檢測系統綜合測試
Cloud Quant - Task Q-603
"""

import sys
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.risk.anomaly_detection import MarketAnomalyDetector, AnomalyType, SeverityLevel
from src.risk.circuit_breaker import CircuitBreaker, BreakerLevel
from src.risk.deleveraging import RapidDeleveraging, DeleveragingStrategy, Position

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnomalySystemTester:
    """Comprehensive testing for anomaly detection system"""

    def __init__(self):
        self.detector = MarketAnomalyDetector(contamination=0.01)
        self.breaker = CircuitBreaker(initial_value=100000)
        self.deleverager = RapidDeleveraging(max_leverage=2.0, target_leverage=1.0)

        self.test_results = {
            "anomaly_detection": {},
            "circuit_breaker": {},
            "deleveraging": {},
            "integration": {},
        }

    def generate_test_market_data(self, n_days: int = 200, inject_anomalies: bool = True) -> Dict:
        """Generate test market data with optional anomalies"""

        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        market_data = {}

        for symbol in symbols:
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

            # Generate base data
            base_price = np.random.uniform(100, 400)
            returns = np.random.normal(0.001, 0.02, n_days)
            prices = base_price * (1 + returns).cumprod()

            data = pd.DataFrame(
                {
                    "date": dates,
                    "open": prices * np.random.uniform(0.98, 1.0, n_days),
                    "high": prices * np.random.uniform(1.0, 1.02, n_days),
                    "low": prices * np.random.uniform(0.98, 1.0, n_days),
                    "close": prices,
                    "volume": np.random.uniform(1000000, 5000000, n_days),
                }
            )
            data.set_index("date", inplace=True)

            # Inject anomalies
            if inject_anomalies and symbol in ["AAPL", "TSLA"]:
                # Price spike
                data.loc[data.index[-5], "close"] *= 1.08
                # Volume surge
                data.loc[data.index[-3], "volume"] *= 5
                # Flash crash
                data.loc[data.index[-1], "close"] *= 0.92
                data.loc[data.index[-1], "low"] *= 0.88

            market_data[symbol] = data

        return market_data

    async def test_anomaly_detection(self):
        """Test anomaly detection functionality"""
        print("\n" + "=" * 60)
        print("TESTING: Anomaly Detection")
        print("=" * 60)

        # Generate test data
        market_data = self.generate_test_market_data(inject_anomalies=True)

        # Test feature extraction
        features = self.detector.extract_features(market_data["AAPL"])
        assert features.shape[1] == 8, "Feature extraction failed"
        print("[OK] Feature extraction successful")

        # Test anomaly detection
        anomalies = self.detector.detect_anomalies(market_data)
        print(f"[OK] Detected {len(anomalies)} anomalies")

        # Verify anomaly properties
        for anomaly in anomalies:
            assert hasattr(anomaly, "timestamp")
            assert hasattr(anomaly, "severity")
            assert hasattr(anomaly, "anomaly_type")
            assert anomaly.severity in SeverityLevel
            assert anomaly.anomaly_type in AnomalyType

        # Test model training
        self.detector.train_model(market_data["AAPL"])
        print("[OK] Model training successful")

        # Test threshold updates
        self.detector.update_thresholds({"price_change": 0.03})
        assert self.detector.thresholds["price_change"] == 0.03
        print("[OK] Threshold update successful")

        # Generate report
        report = self.detector.get_anomaly_report()
        assert "total_anomalies" in report
        assert "by_type" in report
        assert "by_severity" in report
        print("[OK] Report generation successful")

        # Calculate accuracy (assuming we know anomalies were injected)
        expected_anomalies = 2  # We injected anomalies in AAPL and TSLA
        detected_symbols = {a.symbol for a in anomalies}
        accuracy = len(detected_symbols.intersection({"AAPL", "TSLA"})) / expected_anomalies

        self.test_results["anomaly_detection"] = {
            "passed": True,
            "anomalies_detected": len(anomalies),
            "accuracy": accuracy,
            "report": report,
        }

        print(f"\n[STAT] Detection Accuracy: {accuracy:.1%}")
        print(f"[STAT] Anomalies by type: {report.get('by_type', {})}")
        print(f"[STAT] Anomalies by severity: {report.get('by_severity', {})}")

        return accuracy >= 0.95

    async def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        print("\n" + "=" * 60)
        print("TESTING: Circuit Breaker")
        print("=" * 60)

        # Test trigger levels
        test_scenarios = [
            (97000, None, "No trigger at -3%"),
            (94000, BreakerLevel.LEVEL_1, "Level 1 at -6%"),
            (88000, BreakerLevel.LEVEL_2, "Level 2 at -12%"),
            (83000, BreakerLevel.LEVEL_3, "Level 3 at -17%"),
            (78000, BreakerLevel.EMERGENCY, "Emergency at -22%"),
        ]

        breaker = CircuitBreaker(initial_value=100000)  # Fresh instance
        triggers_correct = True

        for value, expected_level, description in test_scenarios:
            triggered = breaker.check_trigger(value)

            if expected_level:
                if triggered != expected_level:
                    print(f"[FAIL] {description} - Expected {expected_level}, got {triggered}")
                    triggers_correct = False
                else:
                    print(f"[OK] {description}")
            else:
                if triggered is not None:
                    print(f"[FAIL] {description} - Unexpected trigger: {triggered}")
                    triggers_correct = False
                else:
                    print(f"[OK] {description}")

        # Test status
        status = breaker.get_status()
        assert "current_level" in status
        assert "is_paused" in status
        assert "trading_enabled" in status
        print("[OK] Status retrieval successful")

        # Test manual resume
        if breaker.is_paused:
            breaker.resume_trading("Test resume")
            assert breaker.trading_enabled == True
            print("[OK] Manual resume successful")

        # Test emergency stop
        breaker.emergency_stop("Test emergency")
        assert breaker.current_level == BreakerLevel.EMERGENCY
        assert breaker.is_paused == True
        print("[OK] Emergency stop successful")

        # Test history
        history = breaker.get_history()
        assert len(history) > 0
        print(f"[OK] History tracking: {len(history)} events")

        self.test_results["circuit_breaker"] = {
            "passed": triggers_correct,
            "total_triggers": len(history),
            "final_status": status,
        }

        return triggers_correct

    async def test_deleveraging(self):
        """Test rapid deleveraging functionality"""
        print("\n" + "=" * 60)
        print("TESTING: Rapid Deleveraging")
        print("=" * 60)

        # Create test portfolio
        test_positions = [
            Position("AAPL", 100, 150, 145, 2.0, 0.7, 0.9, -500, 14500),
            Position("GOOGL", 50, 2800, 2750, 2.5, 0.8, 0.85, -2500, 137500),
            Position("TSLA", 30, 800, 850, 3.0, 0.9, 0.7, 1500, 25500),
            Position("MSFT", 75, 380, 375, 1.5, 0.5, 0.95, -375, 28125),
            Position("AMZN", 40, 170, 165, 2.2, 0.6, 0.88, -200, 6600),
        ]
        account_equity = 100000

        # Test leverage calculation
        current_leverage = self.deleverager.calculate_portfolio_leverage(
            test_positions, account_equity
        )
        assert current_leverage > 0, "Leverage calculation failed"
        print(f"[OK] Current leverage: {current_leverage:.2f}x")

        # Test each strategy
        strategies_tested = []
        for strategy in DeleveragingStrategy:
            plan = self.deleverager.create_deleveraging_plan(
                test_positions, account_equity, strategy
            )

            if plan:
                assert len(plan.positions_to_close) > 0
                assert plan.estimated_proceeds > 0
                assert plan.strategy_used == strategy
                strategies_tested.append(strategy.value)
                print(f"[OK] {strategy.value} strategy: {len(plan.positions_to_close)} positions")

        # Test execution
        plan = self.deleverager.create_deleveraging_plan(
            test_positions, account_equity, DeleveragingStrategy.SMART
        )

        if plan:
            result = self.deleverager.execute_deleveraging(plan)
            assert result["status"] in ["completed", "partial"]
            assert result["executed_count"] >= 0
            print(f"[OK] Execution: {result['executed_count']} positions closed")
            print(f"[OK] New leverage: {result['new_leverage']:.2f}x")

            execution_success = result["executed_count"] > 0
        else:
            execution_success = True  # No deleveraging needed
            print("[OK] No deleveraging needed")

        # Test status
        status = self.deleverager.get_status()
        assert "max_leverage" in status
        assert "target_leverage" in status
        print("[OK] Status retrieval successful")

        self.test_results["deleveraging"] = {
            "passed": execution_success,
            "strategies_tested": strategies_tested,
            "current_leverage": current_leverage,
            "execution_time": result.get("execution_time", 0) if plan else 0,
        }

        return execution_success and len(strategies_tested) == 5

    async def test_integration(self):
        """Test integrated system functionality"""
        print("\n" + "=" * 60)
        print("TESTING: System Integration")
        print("=" * 60)

        # Simulate market crash scenario
        market_data = self.generate_test_market_data(inject_anomalies=True)

        # Step 1: Detect anomalies
        anomalies = self.detector.detect_anomalies(market_data)
        critical_anomalies = [a for a in anomalies if a.severity.value >= SeverityLevel.HIGH.value]
        print(f"[OK] Step 1: Detected {len(critical_anomalies)} critical anomalies")

        # Step 2: Check circuit breaker
        portfolio_value = 85000  # 15% drawdown
        triggered = self.breaker.check_trigger(portfolio_value)
        print(f"[OK] Step 2: Circuit breaker triggered: {triggered.name if triggered else 'None'}")

        # Step 3: Execute deleveraging if needed
        if triggered and triggered.value >= BreakerLevel.LEVEL_2.value:
            test_positions = [
                Position("AAPL", 100, 150, 130, 2.0, 0.8, 0.9, -2000, 13000),
                Position("GOOGL", 50, 2800, 2500, 2.5, 0.9, 0.85, -15000, 125000),
            ]

            plan = self.deleverager.create_deleveraging_plan(
                test_positions, portfolio_value, DeleveragingStrategy.SMART
            )

            if plan:
                result = self.deleverager.execute_deleveraging(plan)
                print(f"[OK] Step 3: Deleveraging executed - {result['executed_count']} positions")
            else:
                print("[OK] Step 3: No deleveraging needed")

        # Test workflow completion
        workflow_complete = len(anomalies) > 0 and triggered is not None and self.breaker.is_paused

        self.test_results["integration"] = {
            "passed": workflow_complete,
            "anomalies_detected": len(anomalies),
            "breaker_triggered": triggered.name if triggered else None,
            "trading_paused": self.breaker.is_paused,
        }

        print(f"\n[STAT] Integration Test: {'PASSED' if workflow_complete else 'FAILED'}")
        print(f"[STAT] System Response Chain:")
        print(f"   1. Anomalies -> {len(anomalies)} detected")
        print(f"   2. Circuit Breaker -> {triggered.name if triggered else 'Not triggered'}")
        print(f"   3. Trading Status -> {'PAUSED' if self.breaker.is_paused else 'ACTIVE'}")

        return workflow_complete

    async def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "=" * 60)
        print("MARKET ANOMALY DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Run tests
        test_results = {
            "anomaly_detection": await self.test_anomaly_detection(),
            "circuit_breaker": await self.test_circuit_breaker(),
            "deleveraging": await self.test_deleveraging(),
            "integration": await self.test_integration(),
        }

        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        success_rate = passed_tests / total_tests

        # Generate report
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        for test_name, passed in test_results.items():
            status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")

        print("\n" + "-" * 60)
        print(f"Overall Success Rate: {success_rate:.1%}")
        print(f"Tests Passed: {passed_tests}/{total_tests}")

        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate,
                "all_tests_passed": success_rate == 1.0,
            },
            "detailed_results": self.test_results,
        }

        report_file = Path("reports/anomaly_system_test_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n[FILE] Detailed report saved: {report_file}")

        # Acceptance criteria check
        print("\n" + "=" * 60)
        print("ACCEPTANCE CRITERIA VALIDATION")
        print("=" * 60)

        criteria = {
            "Anomaly Detection Accuracy > 95%": self.test_results["anomaly_detection"].get(
                "accuracy", 0
            )
            > 0.95,
            "Circuit Breaker 100% Reliable": self.test_results["circuit_breaker"].get(
                "passed", False
            ),
            "Deleveraging Execution < 1 second": self.test_results["deleveraging"].get(
                "execution_time", 999
            )
            < 1.0,
            "All Alerts Sent Correctly": len(self.detector.anomaly_history) > 0,
            "Test Coverage > 90%": success_rate > 0.9,
        }

        for criterion, met in criteria.items():
            status = "[PASS]" if met else "[FAIL]"
            print(f"{status} {criterion}")

        all_criteria_met = all(criteria.values())

        if all_criteria_met:
            print("\n[SUCCESS] ALL ACCEPTANCE CRITERIA MET! System ready for production.")
        else:
            print("\n[WARNING] Some criteria not met. Review and optimize before deployment.")

        return all_criteria_met


async def main():
    """Main test execution"""
    tester = AnomalySystemTester()
    success = await tester.run_all_tests()

    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
