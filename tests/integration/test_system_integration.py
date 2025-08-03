"""
System Integration Tests
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.integration.main_controller import MainController
from src.integration.data_pipeline import DataPipeline
from src.integration.health_monitor import HealthMonitor, HealthStatus


class TestMainController(unittest.TestCase):
    """Test cases for Main Controller"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'mode': 'backtest',
            'symbols': ['AAPL'],
            'initial_capital': 100000,
            'risk_limit': 0.02,
            'data_lookback_days': 30,
            'prediction_horizons': [1, 5],
            'model_paths': {
                'lstm': './test_models/lstm.h5',
                'rl_agent': './test_models/rl_agent'
            }
        }
        
        # Create controller with test config
        self.controller = MainController()
        self.controller.config = self.config
    
    @patch('src.integration.main_controller.CapitalComClient')
    @patch('src.integration.main_controller.LSTMPredictor')
    @patch('src.integration.main_controller.FinBERTAnalyzer')
    @patch('src.integration.main_controller.PPOAgent')
    async def test_initialize_components(self, mock_ppo, mock_finbert, mock_lstm, mock_client):
        """Test component initialization"""
        # Mock client connection
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        
        # Initialize components
        await self.controller.initialize_components()
        
        # Verify all components are initialized
        self.assertIsNotNone(self.controller.data_client)
        self.assertIsNotNone(self.controller.lstm_predictor)
        self.assertIsNotNone(self.controller.sentiment_analyzer)
        self.assertIsNotNone(self.controller.rl_agent)
        self.assertIsNotNone(self.controller.data_pipeline)
        self.assertIsNotNone(self.controller.health_monitor)
        
        # Verify client connection was called
        mock_client_instance.connect.assert_called_once()
    
    async def test_process_trading_signal(self):
        """Test trading signal processing"""
        # Mock components
        self.controller.data_pipeline = Mock()
        self.controller.lstm_predictor = Mock()
        self.controller.sentiment_analyzer = Mock()
        self.controller.rl_agent = Mock()
        
        # Mock data
        market_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        self.controller.data_pipeline.prepare_features.return_value = market_data
        self.controller.lstm_predictor.predict.return_value = 0.02  # 2% predicted return
        self.controller.rl_agent.predict.return_value = (1, 0.8)  # BUY action with 80% confidence
        
        # Mock async methods
        self.controller._get_lstm_predictions = AsyncMock(return_value={'horizon_1d': 0.02})
        self.controller._get_sentiment_analysis = AsyncMock(return_value={'overall_sentiment': 0.5})
        self.controller._execute_trade = AsyncMock(return_value={'success': True, 'pnl': 100})
        
        # Process signal
        await self.controller._process_trading_signal(
            symbol='AAPL',
            timestamp=datetime.now(),
            data=market_data
        )
        
        # Verify methods were called
        self.controller.data_pipeline.prepare_features.assert_called_once()
        self.controller.rl_agent.predict.assert_called_once()
        self.controller._execute_trade.assert_called_once()
        
        # Verify performance metrics updated
        self.assertEqual(self.controller.performance_metrics['total_trades'], 1)
        self.assertEqual(self.controller.performance_metrics['successful_trades'], 1)
        self.assertEqual(self.controller.performance_metrics['total_pnl'], 100)
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        self.controller.config['risk_limit'] = 0.02
        self.controller.config['initial_capital'] = 100000
        
        # Test position sizing
        position_size = self.controller._calculate_position_size(
            symbol='AAPL',
            size_pct=0.5,
            confidence=0.8
        )
        
        # Expected: 100000 * 0.02 * 0.5 * 0.8 / 100 = 8 shares
        self.assertEqual(position_size, 8)
    
    def test_load_config(self):
        """Test configuration loading"""
        # Test default config
        config = self.controller._load_config(None)
        self.assertIn('mode', config)
        self.assertIn('symbols', config)
        self.assertIn('initial_capital', config)
        
        # Test custom config file
        test_config = {'mode': 'paper', 'symbols': ['TSLA']}
        test_path = Path('test_config.json')
        
        with open(test_path, 'w') as f:
            json.dump(test_config, f)
        
        try:
            config = self.controller._load_config(str(test_path))
            self.assertEqual(config['mode'], 'paper')
            self.assertEqual(config['symbols'], ['TSLA'])
        finally:
            test_path.unlink()


class TestDataPipeline(unittest.TestCase):
    """Test cases for Data Pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_client = AsyncMock()
        self.symbols = ['AAPL', 'GOOGL']
        self.pipeline = DataPipeline(
            symbols=self.symbols,
            data_client=self.mock_client,
            buffer_size=100
        )
    
    async def test_get_historical_data(self):
        """Test historical data fetching"""
        # Mock client response
        mock_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        self.mock_client.get_historical_data.return_value = mock_data
        
        # Mock processing
        self.pipeline._process_historical_data = AsyncMock(return_value=mock_data)
        
        # Get historical data
        result = await self.pipeline.get_historical_data(
            symbol='AAPL',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3)
        )
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.mock_client.get_historical_data.assert_called_once()
    
    def test_process_market_data(self):
        """Test market data processing"""
        # Prepare test data
        raw_data = {
            'timestamp': datetime.now(),
            'open': '100.5',
            'high': '101.0',
            'low': '99.5',
            'close': '100.8',
            'volume': '1000000'
        }
        
        # Process data
        processed = self.pipeline._process_market_data('AAPL', raw_data)
        
        # Verify processing
        self.assertEqual(processed['symbol'], 'AAPL')
        self.assertEqual(processed['close'], 100.8)
        self.assertEqual(processed['volume'], 1000000.0)
        self.assertIsInstance(processed['timestamp'], datetime)
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation"""
        # Create test DataFrame
        df = pd.DataFrame({
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(900000, 1100000, 100)
        })
        
        # Calculate indicators
        indicators = self.pipeline._calculate_technical_indicators(df)
        
        # Verify indicators exist
        self.assertIn('SMA_20', indicators)
        self.assertIn('SMA_50', indicators)
        self.assertIn('RSI', indicators)
        self.assertIn('MACD', indicators)
        self.assertIn('BB_upper', indicators)
        self.assertIn('BB_lower', indicators)
        self.assertIn('ATR', indicators)
        self.assertIn('Volume_MA', indicators)
        
        # Verify values are numeric
        for value in indicators.values():
            self.assertTrue(isinstance(value, (int, float, np.number)))
    
    def test_data_quality_metrics(self):
        """Test data quality metrics tracking"""
        # Simulate data processing
        self.pipeline.data_quality_metrics['total_received'] = 100
        self.pipeline.data_quality_metrics['processed'] = 95
        self.pipeline.data_quality_metrics['errors'] = 5
        self.pipeline.data_quality_metrics['latency_ms'].extend([10, 20, 30, 40, 50])
        
        # Get metrics
        metrics = self.pipeline.get_data_quality_metrics()
        
        # Verify metrics
        self.assertEqual(metrics['total_received'], 100)
        self.assertEqual(metrics['processed'], 95)
        self.assertEqual(metrics['errors'], 5)
        self.assertEqual(metrics['success_rate'], 0.95)
        self.assertEqual(metrics['avg_latency_ms'], 30)


class TestHealthMonitor(unittest.TestCase):
    """Test cases for Health Monitor"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_components = {
            'data_client': Mock(),
            'lstm_predictor': Mock(),
            'sentiment_analyzer': Mock(),
            'rl_agent': Mock(),
            'data_pipeline': Mock()
        }
        
        self.monitor = HealthMonitor(
            components=self.mock_components,
            check_interval=1
        )
    
    async def test_check_all_components(self):
        """Test checking all components"""
        # Mock component methods
        self.mock_components['data_client'].is_connected = AsyncMock(return_value=True)
        self.mock_components['data_client'].last_data_update = datetime.now()
        
        # Check components
        health_report = await self.monitor.check_all_components()
        
        # Verify report
        self.assertIn('data_client', health_report)
        self.assertIsInstance(health_report['data_client'].status, HealthStatus)
        self.assertIsInstance(health_report['data_client'].last_check, datetime)
    
    def test_check_system_resources(self):
        """Test system resource checking"""
        # Check resources
        resource_report = self.monitor.check_system_resources()
        
        # Verify report structure
        self.assertIn('cpu', resource_report)
        self.assertIn('memory', resource_report)
        self.assertIn('disk', resource_report)
        
        # Verify CPU metrics
        self.assertIn('percent', resource_report['cpu'])
        self.assertIn('status', resource_report['cpu'])
        self.assertIn('cores', resource_report['cpu'])
        
        # Verify values are reasonable
        self.assertGreaterEqual(resource_report['cpu']['percent'], 0)
        self.assertLessEqual(resource_report['cpu']['percent'], 100)
    
    def test_determine_overall_status(self):
        """Test overall status determination"""
        from src.integration.health_monitor import ComponentHealth
        
        # Test all healthy
        component_health = {
            'comp1': ComponentHealth('comp1', HealthStatus.HEALTHY, datetime.now(), 'OK', {}),
            'comp2': ComponentHealth('comp2', HealthStatus.HEALTHY, datetime.now(), 'OK', {})
        }
        resource_report = {
            'cpu': {'status': 'healthy'},
            'memory': {'status': 'healthy'}
        }
        
        status = self.monitor._determine_overall_status(component_health, resource_report)
        self.assertEqual(status, HealthStatus.HEALTHY)
        
        # Test with warning
        component_health['comp1'].status = HealthStatus.WARNING
        status = self.monitor._determine_overall_status(component_health, resource_report)
        self.assertEqual(status, HealthStatus.WARNING)
        
        # Test with critical
        component_health['comp2'].status = HealthStatus.CRITICAL
        status = self.monitor._determine_overall_status(component_health, resource_report)
        self.assertEqual(status, HealthStatus.CRITICAL)
        
        # Test with down
        component_health['comp1'].status = HealthStatus.DOWN
        status = self.monitor._determine_overall_status(component_health, resource_report)
        self.assertEqual(status, HealthStatus.DOWN)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    @patch('src.integration.main_controller.CapitalComClient')
    @patch('src.integration.main_controller.LSTMPredictor')
    @patch('src.integration.main_controller.FinBERTAnalyzer')
    @patch('src.integration.main_controller.PPOAgent')
    async def test_full_system_flow(self, mock_ppo, mock_finbert, mock_lstm, mock_client):
        """Test full system integration flow"""
        # Create controller
        controller = MainController()
        controller.config['mode'] = 'backtest'
        controller.config['symbols'] = ['AAPL']
        
        # Mock client
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        
        # Mock historical data
        historical_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        # Initialize components
        await controller.initialize_components()
        
        # Mock data pipeline method
        controller.data_pipeline.get_historical_data = AsyncMock(return_value=historical_data)
        
        # Mock predictions
        controller._get_lstm_predictions = AsyncMock(return_value={'horizon_1d': 0.02})
        controller._get_sentiment_analysis = AsyncMock(return_value={'overall_sentiment': 0.5})
        
        # Mock RL agent
        mock_ppo_instance = controller.rl_agent
        mock_ppo_instance.predict.return_value = (1, 0.8)  # BUY with 80% confidence
        
        # Mock backtester
        controller.backtester = Mock()
        controller.backtester.place_order.return_value = {'order_id': '123', 'pnl': 50}
        
        # Run backtest (simplified)
        await controller._run_backtest()
        
        # Verify data pipeline was called
        controller.data_pipeline.get_historical_data.assert_called()
        
        # Verify predictions were made
        self.assertGreater(controller._get_lstm_predictions.call_count, 0)
        
        # Verify trades were attempted
        self.assertGreater(controller.performance_metrics['total_trades'], 0)


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    # Run tests
    unittest.main()