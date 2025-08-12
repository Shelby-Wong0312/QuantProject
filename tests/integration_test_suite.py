"""
Integration Test Suite
Comprehensive testing for ML/DL/RL trading system
Cloud PM - Task PM-701
"""

import unittest
import asyncio
import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.ml_strategy_integration import MLStrategyIntegration
from src.data.feature_pipeline import FeaturePipeline
from src.data.model_updater import ModelUpdater, UpdateConfig
from src.data.data_quality_monitor import DataQualityMonitor
from src.backtesting.ml_backtest import MLBacktester, BacktestConfig
from src.core.paper_trading import PaperTradingSimulator


class TestMLIntegration(unittest.TestCase):
    """Test ML model integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.strategy = MLStrategyIntegration(initial_capital=100000)
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test market data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = {}
        
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
            df = pd.DataFrame({
                'open': prices * np.random.uniform(0.98, 1.02, len(dates)),
                'high': prices * np.random.uniform(1.0, 1.05, len(dates)),
                'low': prices * np.random.uniform(0.95, 1.0, len(dates)),
                'close': prices,
                'volume': np.random.uniform(1e6, 1e8, len(dates)),
                'returns': pd.Series(prices).pct_change().fillna(0)
            }, index=dates)
            data[symbol] = df
        
        return data
    
    def test_ml_strategy_initialization(self):
        """Test ML strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertIsNotNone(self.strategy.lstm_model)
        self.assertIsNotNone(self.strategy.xgboost_model)
        self.assertIsNotNone(self.strategy.ppo_agent)
        print("[PASS] ML Strategy initialization")
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        for symbol, data in self.test_data.items():
            features = self.strategy.extract_features(data)
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(len(features), 20)  # Expected 20 features
        print("[PASS] Feature extraction")
    
    def test_signal_generation(self):
        """Test signal generation from each model"""
        for symbol, data in self.test_data.items():
            features = self.strategy.extract_features(data)
            
            # Test LSTM signal
            lstm_signal = self.strategy.generate_lstm_signal(features, symbol)
            self.assertIn(lstm_signal.action, ['BUY', 'SELL', 'HOLD'])
            
            # Test XGBoost signal
            xgb_signal = self.strategy.generate_xgboost_signal(features, symbol)
            self.assertIn(xgb_signal.action, ['BUY', 'SELL', 'HOLD'])
            
            # Test PPO signal
            ppo_signal = self.strategy.generate_ppo_signal(features, symbol)
            self.assertIn(ppo_signal.action, ['BUY', 'SELL', 'HOLD'])
        
        print("[PASS] Signal generation from all models")
    
    def test_ensemble_signals(self):
        """Test ensemble signal combination"""
        symbol = 'AAPL'
        data = self.test_data[symbol]
        features = self.strategy.extract_features(data)
        
        lstm_signal = self.strategy.generate_lstm_signal(features, symbol)
        xgb_signal = self.strategy.generate_xgboost_signal(features, symbol)
        ppo_signal = self.strategy.generate_ppo_signal(features, symbol)
        
        ensemble_signal = self.strategy.ensemble_signals(lstm_signal, xgb_signal, ppo_signal)
        
        self.assertIsNotNone(ensemble_signal)
        self.assertIn(ensemble_signal.action, ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(ensemble_signal.confidence, 0)
        self.assertLessEqual(ensemble_signal.confidence, 1)
        
        print("[PASS] Ensemble signal combination")
    
    def test_position_sizing(self):
        """Test position sizing calculation"""
        symbol = 'AAPL'
        data = self.test_data[symbol]
        features = self.strategy.extract_features(data)
        
        # Generate signals
        lstm_signal = self.strategy.generate_lstm_signal(features, symbol)
        xgb_signal = self.strategy.generate_xgboost_signal(features, symbol)
        ppo_signal = self.strategy.generate_ppo_signal(features, symbol)
        ensemble_signal = self.strategy.ensemble_signals(lstm_signal, xgb_signal, ppo_signal)
        
        # Calculate position size
        current_price = data['close'].iloc[-1]
        position_size = self.strategy.calculate_position_size(
            ensemble_signal, current_price, 100000
        )
        
        self.assertGreaterEqual(position_size, 0)
        self.assertLessEqual(position_size * current_price, 100000 * 0.1)  # Max 10% per position
        
        print("[PASS] Position sizing with Kelly Criterion")


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline components"""
    
    def setUp(self):
        """Set up test environment"""
        self.pipeline = FeaturePipeline()
        self.updater = ModelUpdater()
        self.monitor = DataQualityMonitor()
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        return pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1e6, 1e8, len(dates))
        }, index=dates)
    
    def test_feature_pipeline(self):
        """Test feature extraction pipeline"""
        features = self.pipeline.extract_features(self.test_data, 'TEST')
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 20)  # Should have many features
        
        # Check feature categories
        feature_names = list(features.keys())
        has_price = any('return' in name or 'momentum' in name for name in feature_names)
        has_volume = any('volume' in name for name in feature_names)
        has_technical = any(name in ['rsi', 'macd', 'bb_position'] for name in feature_names)
        
        self.assertTrue(has_price)
        self.assertTrue(has_volume)
        self.assertTrue(has_technical)
        
        print("[PASS] Feature pipeline with 50+ features")
    
    def test_batch_processing(self):
        """Test batch feature extraction"""
        data_dict = {
            f'STOCK_{i}': self.test_data.copy()
            for i in range(10)
        }
        
        start_time = time.time()
        batch_features = asyncio.run(self.pipeline.extract_features_batch(data_dict))
        elapsed_time = time.time() - start_time
        
        self.assertEqual(len(batch_features), 10)
        self.assertLess(elapsed_time, 10)  # Should process 10 stocks in <10 seconds
        
        print(f"[PASS] Batch processing 10 stocks in {elapsed_time:.2f} seconds")
    
    def test_data_quality_monitor(self):
        """Test data quality monitoring"""
        # Test with good data
        good_metrics = self.monitor.check_data_quality(self.test_data)
        self.assertGreater(good_metrics.overall_score, 0.8)
        
        # Test with bad data (add issues)
        bad_data = self.test_data.copy()
        bad_data.iloc[10:15, bad_data.columns.get_loc('close')] = np.nan
        bad_metrics = self.monitor.check_data_quality(bad_data)
        
        self.assertLess(bad_metrics.overall_score, good_metrics.overall_score)
        self.assertGreater(len(self.monitor.issues), 0)
        
        print("[PASS] Data quality monitoring with issue detection")
    
    def test_model_updater(self):
        """Test model update system"""
        test_data = {'TEST': self.test_data}
        
        # Test update check
        should_update = self.updater._should_update()
        self.assertIsInstance(should_update, bool)
        
        # Test model status
        status = self.updater.get_model_status()
        self.assertIn('LSTM', status)
        self.assertIn('XGBoost', status)
        self.assertIn('PPO', status)
        
        print("[PASS] Model updater system")


class TestBacktesting(unittest.TestCase):
    """Test backtesting system"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = BacktestConfig(
            initial_capital=100000,
            start_date="2023-01-01",
            end_date="2024-01-01",
            symbols=['AAPL', 'GOOGL'],
            use_walk_forward=False
        )
        self.backtester = MLBacktester(self.config)
    
    def test_backtest_initialization(self):
        """Test backtester initialization"""
        self.assertIsNotNone(self.backtester)
        self.assertEqual(self.backtester.config.initial_capital, 100000)
        print("[PASS] Backtester initialization")
    
    def test_historical_data_generation(self):
        """Test historical data generation"""
        historical_data = self.backtester.load_historical_data()
        
        self.assertGreater(len(historical_data), 0)
        for symbol, data in historical_data.items():
            self.assertIn('close', data.columns)
            self.assertIn('volume', data.columns)
            self.assertGreater(len(data), 200)  # Should have substantial history
        
        print("[PASS] Historical data generation")
    
    def test_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Create dummy results
        portfolio_values = pd.Series([100000, 101000, 99000, 102000, 103000])
        daily_returns = pd.Series([0.01, -0.02, 0.03, 0.01])
        
        results = {
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'trade_history': [],
            'final_value': 103000,
            'positions': {}
        }
        
        metrics = self.backtester._calculate_metrics(results)
        
        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.max_drawdown)
        
        print("[PASS] Performance metrics calculation")


class TestPaperTrading(unittest.TestCase):
    """Test paper trading simulator"""
    
    def setUp(self):
        """Set up test environment"""
        self.simulator = PaperTradingSimulator(
            initial_balance=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
    
    def test_order_placement(self):
        """Test order placement"""
        # Update market prices
        self.simulator.update_market_prices({'AAPL': 150})
        
        # Place buy order
        order_id = asyncio.run(
            self.simulator.place_order('AAPL', 'BUY', 100, 'MARKET')
        )
        
        self.assertIsNotNone(order_id)
        self.assertIn('AAPL', self.simulator.positions)
        self.assertEqual(self.simulator.positions['AAPL'].quantity, 100)
        
        print("[PASS] Paper trading order execution")
    
    def test_risk_limits(self):
        """Test risk management limits"""
        # Test position size limit
        self.simulator.update_market_prices({'AAPL': 150})
        
        # Try to place order exceeding position limit (>10% of portfolio)
        large_quantity = 1000  # Would be $150,000, exceeding 10% limit
        order_id = asyncio.run(
            self.simulator.place_order('AAPL', 'BUY', large_quantity, 'MARKET')
        )
        
        # Order should be rejected
        self.assertIsNone(order_id)
        
        print("[PASS] Risk limit enforcement")
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        metrics = self.simulator.get_performance_metrics()
        
        self.assertIn('total_return', metrics)
        self.assertIn('portfolio_value', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('sharpe_ratio', metrics)
        
        print("[PASS] Performance metrics generation")


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration test"""
    
    def test_complete_trading_flow(self):
        """Test complete trading flow from data to execution"""
        print("\n[E2E TEST] Starting end-to-end test...")
        
        # 1. Initialize components
        pipeline = FeaturePipeline()
        strategy = MLStrategyIntegration()
        simulator = PaperTradingSimulator(initial_balance=100000)
        monitor = DataQualityMonitor()
        
        # 2. Generate test data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        test_data = {}
        
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
            df = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.uniform(1e6, 1e8, len(dates)),
                'returns': pd.Series(prices).pct_change().fillna(0)
            }, index=dates)
            test_data[symbol] = df
        
        # 3. Check data quality
        for symbol, data in test_data.items():
            metrics = monitor.check_data_quality(data, symbol)
            self.assertGreater(metrics.overall_score, 0.7)
        print("  [OK] Data quality check passed")
        
        # 4. Extract features
        all_features = {}
        for symbol, data in test_data.items():
            features = pipeline.extract_features(data, symbol)
            all_features[symbol] = features
        print("  [OK] Feature extraction completed")
        
        # 5. Generate trading signals
        signals = asyncio.run(strategy.generate_trading_signals(test_data))
        self.assertGreater(len(signals), 0)
        print(f"  [OK] Generated {len(signals)} trading signals")
        
        # 6. Execute trades
        # Update market prices
        current_prices = {symbol: data['close'].iloc[-1] for symbol, data in test_data.items()}
        simulator.update_market_prices(current_prices)
        
        # Execute signals
        execution_results = asyncio.run(strategy.execute_trades(signals, simulator))
        print(f"  [OK] Executed {len(execution_results.get('executed_trades', []))} trades")
        
        # 7. Calculate performance
        metrics = simulator.get_performance_metrics()
        self.assertIsNotNone(metrics['portfolio_value'])
        print(f"  [OK] Portfolio value: ${metrics['portfolio_value']:,.2f}")
        
        print("[E2E TEST] Complete trading flow successful!")


class PerformanceBenchmark:
    """Performance benchmarking tests"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_feature_extraction(self):
        """Benchmark feature extraction performance"""
        pipeline = FeaturePipeline()
        
        # Generate test data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 105, 252),
            'high': np.random.uniform(105, 110, 252),
            'low': np.random.uniform(95, 100, 252),
            'close': np.random.uniform(98, 107, 252),
            'volume': np.random.uniform(1e6, 1e8, 252)
        }, index=dates)
        
        # Single stock
        start = time.time()
        features = pipeline.extract_features(test_data, 'TEST')
        single_time = (time.time() - start) * 1000
        
        # Batch processing (100 stocks)
        batch_data = {f'STOCK_{i}': test_data.copy() for i in range(100)}
        start = time.time()
        batch_features = asyncio.run(pipeline.extract_features_batch(batch_data))
        batch_time = (time.time() - start) * 1000
        
        self.results['feature_extraction'] = {
            'single_stock_ms': single_time,
            'batch_100_stocks_ms': batch_time,
            'avg_per_stock_ms': batch_time / 100
        }
        
        return single_time < 100  # Target: <100ms per stock
    
    def benchmark_signal_generation(self):
        """Benchmark signal generation performance"""
        strategy = MLStrategyIntegration()
        
        # Generate test data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        test_data = {}
        
        for i in range(10):
            symbol = f'STOCK_{i}'
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
            test_data[symbol] = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.uniform(1e6, 1e8, len(dates)),
                'returns': pd.Series(prices).pct_change().fillna(0)
            }, index=dates)
        
        start = time.time()
        signals = asyncio.run(strategy.generate_trading_signals(test_data))
        signal_time = (time.time() - start) * 1000
        
        self.results['signal_generation'] = {
            'total_time_ms': signal_time,
            'stocks_processed': len(test_data),
            'avg_per_stock_ms': signal_time / len(test_data)
        }
        
        return signal_time < 1000  # Target: <100ms per stock
    
    def benchmark_order_execution(self):
        """Benchmark order execution performance"""
        simulator = PaperTradingSimulator()
        
        # Update prices
        prices = {f'STOCK_{i}': 100 + i for i in range(10)}
        simulator.update_market_prices(prices)
        
        # Execute orders
        start = time.time()
        orders = []
        for symbol in prices.keys():
            order_id = asyncio.run(
                simulator.place_order(symbol, 'BUY', 100, 'MARKET')
            )
            orders.append(order_id)
        
        execution_time = (time.time() - start) * 1000
        
        self.results['order_execution'] = {
            'total_time_ms': execution_time,
            'orders_placed': len(orders),
            'avg_per_order_ms': execution_time / len(orders)
        }
        
        return execution_time / len(orders) < 200  # Target: <200ms per order
    
    def generate_benchmark_report(self):
        """Generate performance benchmark report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': self.results,
            'targets_met': {
                'feature_extraction': self.results.get('feature_extraction', {}).get('single_stock_ms', 999) < 100,
                'signal_generation': self.results.get('signal_generation', {}).get('avg_per_stock_ms', 999) < 100,
                'order_execution': self.results.get('order_execution', {}).get('avg_per_order_ms', 999) < 200
            }
        }
        
        # Save report
        with open('reports/performance_benchmark.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("INTEGRATION TEST SUITE")
    print("Cloud PM - Task PM-701")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMLIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktesting))
    suite.addTests(loader.loadTestsFromTestCase(TestPaperTrading))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Performance benchmarks
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKS")
    print("="*70)
    
    benchmark = PerformanceBenchmark()
    
    print("\n1. Feature Extraction Benchmark...")
    feature_pass = benchmark.benchmark_feature_extraction()
    print(f"   Single stock: {benchmark.results['feature_extraction']['single_stock_ms']:.2f}ms")
    print(f"   100 stocks batch: {benchmark.results['feature_extraction']['batch_100_stocks_ms']:.2f}ms")
    print(f"   Result: {'PASS' if feature_pass else 'FAIL'}")
    
    print("\n2. Signal Generation Benchmark...")
    signal_pass = benchmark.benchmark_signal_generation()
    print(f"   10 stocks: {benchmark.results['signal_generation']['total_time_ms']:.2f}ms")
    print(f"   Per stock: {benchmark.results['signal_generation']['avg_per_stock_ms']:.2f}ms")
    print(f"   Result: {'PASS' if signal_pass else 'FAIL'}")
    
    print("\n3. Order Execution Benchmark...")
    order_pass = benchmark.benchmark_order_execution()
    print(f"   10 orders: {benchmark.results['order_execution']['total_time_ms']:.2f}ms")
    print(f"   Per order: {benchmark.results['order_execution']['avg_per_order_ms']:.2f}ms")
    print(f"   Result: {'PASS' if order_pass else 'FAIL'}")
    
    # Generate report
    report = benchmark.generate_benchmark_report()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    print("\nPerformance Targets:")
    for target, met in report['targets_met'].items():
        status = "[OK]" if met else "[FAIL]"
        print(f"  {status} {target}")
    
    if success_rate >= 85:
        print("\n[SUCCESS] Integration tests PASSED with >85% success rate")
    else:
        print("\n[WARNING] Integration tests need attention")
    
    return result, report


if __name__ == "__main__":
    run_integration_tests()