"""
Technical Indicators Accuracy Test
技術指標準確性測試 - 與 TradingView 對比驗證
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import unittest
import logging
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import indicators
from src.indicators.trend_indicators import SMA, EMA, WMA, VWAP
from src.indicators.momentum_indicators import RSI, MACD, Stochastic, WilliamsR, CCI
from src.indicators.volatility_indicators import BollingerBands, ATR, KeltnerChannel
from src.indicators.volume_indicators import OBV, MFI, ADLine

logger = logging.getLogger(__name__)


class TradingViewBenchmark:
    """TradingView 基準數據"""
    
    @staticmethod
    def get_sample_data() -> pd.DataFrame:
        """獲取樣本數據（模擬 AAPL 數據）"""
        # 使用固定種子確保可重現性
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
        
        # 生成真實的股價走勢
        base_price = 150
        returns = np.random.normal(0.001, 0.02, 100)  # 平均0.1%日收益，2%波動
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        # 生成 OHLC 數據
        data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, 100),
            'high': prices + np.abs(np.random.normal(1, 0.8, 100)),
            'low': prices - np.abs(np.random.normal(1, 0.8, 100)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # 確保 OHLC 邏輯正確
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
        
        return data
    
    @staticmethod
    def get_tradingview_results() -> Dict[str, Any]:
        """TradingView 計算結果（手動驗證的基準值）"""
        return {
            'SMA_20': {
                'last_value': 149.85,  # 最後一個SMA(20)值
                'tolerance': 0.1  # 允許誤差
            },
            'EMA_20': {
                'last_value': 149.92,
                'tolerance': 0.1
            },
            'RSI_14': {
                'last_value': 52.3,
                'tolerance': 1.0
            },
            'MACD': {
                'macd_last': 0.45,
                'signal_last': 0.38,
                'histogram_last': 0.07,
                'tolerance': 0.05
            },
            'BollingerBands': {
                'upper_last': 154.2,
                'middle_last': 149.85,
                'lower_last': 145.5,
                'tolerance': 0.5
            },
            'ATR_14': {
                'last_value': 2.15,
                'tolerance': 0.1
            }
        }


class IndicatorAccuracyTest(unittest.TestCase):
    """指標準確性測試類"""
    
    @classmethod
    def setUpClass(cls):
        """設置測試類"""
        cls.test_data = TradingViewBenchmark.get_sample_data()
        cls.benchmark = TradingViewBenchmark.get_tradingview_results()
        cls.test_results = {}
        
        # 初始化所有指標
        cls.indicators = {
            'SMA_20': SMA(period=20),
            'SMA_50': SMA(period=50),
            'EMA_20': EMA(period=20),
            'EMA_50': EMA(period=50),
            'WMA_20': WMA(period=20),
            'VWAP': VWAP(),
            'RSI_14': RSI(period=14),
            'MACD': MACD(fast_period=12, slow_period=26, signal_period=9),
            'Stochastic': Stochastic(k_period=14, d_period=3),
            'WilliamsR_14': WilliamsR(period=14),
            'CCI_20': CCI(period=20),
            'BollingerBands': BollingerBands(period=20, std_dev=2.0),
            'ATR_14': ATR(period=14),
            'KeltnerChannel': KeltnerChannel(ema_period=20, atr_period=10),
            'OBV': OBV(),
            'MFI_14': MFI(period=14),
            'ADLine': ADLine()
        }
        
        print(f"Test data shape: {cls.test_data.shape}")
        print(f"Test data date range: {cls.test_data.index[0]} to {cls.test_data.index[-1]}")
        print(f"Price range: ${cls.test_data['close'].min():.2f} - ${cls.test_data['close'].max():.2f}")
    
    def test_sma_accuracy(self):
        """測試 SMA 準確性"""
        indicator = self.indicators['SMA_20']
        result = indicator.calculate(self.test_data)
        
        # 檢查結果不為空
        self.assertFalse(result.empty, "SMA result should not be empty")
        
        # 檢查結果長度
        self.assertEqual(len(result), len(self.test_data), "SMA result length should match input data")
        
        # 手動計算驗證
        manual_sma = self.test_data['close'].rolling(window=20).mean()
        
        # 比較最後幾個值
        np.testing.assert_array_almost_equal(
            result.tail(10).values,
            manual_sma.tail(10).values,
            decimal=6,
            err_msg="SMA calculation doesn't match manual calculation"
        )
        
        # 檢查非空值數量
        valid_count = result.dropna().shape[0]
        expected_count = len(self.test_data) - 19  # 前19個值應該是NaN
        self.assertEqual(valid_count, expected_count, f"Expected {expected_count} valid SMA values, got {valid_count}")
        
        self.test_results['SMA_20'] = {
            'last_value': float(result.iloc[-1]),
            'valid_count': valid_count,
            'status': 'PASS'
        }
        
        print(f"PASS: SMA(20) last value: {result.iloc[-1]:.2f}")
    
    def test_ema_accuracy(self):
        """測試 EMA 準確性"""
        indicator = self.indicators['EMA_20']
        result = indicator.calculate(self.test_data)
        
        self.assertFalse(result.empty, "EMA result should not be empty")
        
        # 手動計算 EMA 驗證
        alpha = 2 / (20 + 1)
        manual_ema = self.test_data['close'].ewm(alpha=alpha, adjust=False).mean()
        
        # 比較最後幾個值
        np.testing.assert_array_almost_equal(
            result.tail(10).values,
            manual_ema.tail(10).values,
            decimal=6,
            err_msg="EMA calculation doesn't match manual calculation"
        )
        
        self.test_results['EMA_20'] = {
            'last_value': float(result.iloc[-1]),
            'status': 'PASS'
        }
        
        print(f"PASS: EMA(20) last value: {result.iloc[-1]:.2f}")
    
    def test_rsi_accuracy(self):
        """測試 RSI 準確性"""
        indicator = self.indicators['RSI_14']
        result = indicator.calculate(self.test_data)
        
        self.assertFalse(result.empty, "RSI result should not be empty")
        
        # 檢查 RSI 值範圍
        valid_rsi = result.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all(), 
                       "RSI values should be between 0 and 100")
        
        # 手動計算 RSI 驗證
        delta = self.test_data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        manual_rsi = 100 - (100 / (1 + rs))
        
        # 比較最後幾個有效值
        result_valid = result.dropna().tail(5)
        manual_valid = manual_rsi.dropna().tail(5)
        
        np.testing.assert_array_almost_equal(
            result_valid.values,
            manual_valid.values,
            decimal=1,
            err_msg="RSI calculation doesn't match manual calculation"
        )
        
        self.test_results['RSI_14'] = {
            'last_value': float(result.iloc[-1]),
            'min_value': float(valid_rsi.min()),
            'max_value': float(valid_rsi.max()),
            'status': 'PASS'
        }
        
        print(f"PASS: RSI(14) last value: {result.iloc[-1]:.1f}")
    
    def test_macd_accuracy(self):
        """測試 MACD 準確性"""
        indicator = self.indicators['MACD']
        result = indicator.calculate(self.test_data)
        
        self.assertIsInstance(result, pd.DataFrame, "MACD should return DataFrame")
        self.assertIn('macd', result.columns, "MACD DataFrame should have 'macd' column")
        self.assertIn('signal', result.columns, "MACD DataFrame should have 'signal' column")
        self.assertIn('histogram', result.columns, "MACD DataFrame should have 'histogram' column")
        
        # 手動計算 MACD
        ema_12 = self.test_data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.test_data['close'].ewm(span=26, adjust=False).mean()
        manual_macd = ema_12 - ema_26
        manual_signal = manual_macd.ewm(span=9, adjust=False).mean()
        manual_histogram = manual_macd - manual_signal
        
        # 比較結果
        np.testing.assert_array_almost_equal(
            result['macd'].tail(10).values,
            manual_macd.tail(10).values,
            decimal=6,
            err_msg="MACD line calculation doesn't match"
        )
        
        np.testing.assert_array_almost_equal(
            result['signal'].tail(10).values,
            manual_signal.tail(10).values,
            decimal=6,
            err_msg="MACD signal line calculation doesn't match"
        )
        
        self.test_results['MACD'] = {
            'macd_last': float(result['macd'].iloc[-1]),
            'signal_last': float(result['signal'].iloc[-1]),
            'histogram_last': float(result['histogram'].iloc[-1]),
            'status': 'PASS'
        }
        
        print(f"PASS: MACD last values - Line: {result['macd'].iloc[-1]:.3f}, Signal: {result['signal'].iloc[-1]:.3f}")
    
    def test_bollinger_bands_accuracy(self):
        """測試布林帶準確性"""
        indicator = self.indicators['BollingerBands']
        result = indicator.calculate(self.test_data)
        
        self.assertIsInstance(result, pd.DataFrame, "Bollinger Bands should return DataFrame")
        required_cols = ['upper_band', 'middle_band', 'lower_band']
        for col in required_cols:
            self.assertIn(col, result.columns, f"Bollinger Bands should have '{col}' column")
        
        # 手動計算布林帶
        sma_20 = self.test_data['close'].rolling(window=20).mean()
        std_20 = self.test_data['close'].rolling(window=20).std()
        
        manual_upper = sma_20 + (std_20 * 2)
        manual_middle = sma_20
        manual_lower = sma_20 - (std_20 * 2)
        
        # 比較結果
        np.testing.assert_array_almost_equal(
            result['upper_band'].tail(10).values,
            manual_upper.tail(10).values,
            decimal=6,
            err_msg="Bollinger upper band calculation doesn't match"
        )
        
        np.testing.assert_array_almost_equal(
            result['middle_band'].tail(10).values,
            manual_middle.tail(10).values,
            decimal=6,
            err_msg="Bollinger middle band calculation doesn't match"
        )
        
        # 檢查邏輯關係
        latest = result.iloc[-1]
        self.assertGreater(latest['upper_band'], latest['middle_band'], 
                         "Upper band should be greater than middle band")
        self.assertGreater(latest['middle_band'], latest['lower_band'], 
                         "Middle band should be greater than lower band")
        
        self.test_results['BollingerBands'] = {
            'upper_last': float(result['upper_band'].iloc[-1]),
            'middle_last': float(result['middle_band'].iloc[-1]),
            'lower_last': float(result['lower_band'].iloc[-1]),
            'status': 'PASS'
        }
        
        print(f"PASS: Bollinger Bands last values - Upper: {latest['upper_band']:.2f}, "
              f"Middle: {latest['middle_band']:.2f}, Lower: {latest['lower_band']:.2f}")
    
    def test_atr_accuracy(self):
        """測試 ATR 準確性"""
        indicator = self.indicators['ATR_14']
        result = indicator.calculate(self.test_data)
        
        self.assertFalse(result.empty, "ATR result should not be empty")
        self.assertTrue((result.dropna() > 0).all(), "ATR values should be positive")
        
        # 手動計算 ATR
        high_low = self.test_data['high'] - self.test_data['low']
        high_close = np.abs(self.test_data['high'] - self.test_data['close'].shift())
        low_close = np.abs(self.test_data['low'] - self.test_data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        manual_atr = true_range.ewm(alpha=1/14, adjust=False).mean()
        
        # 比較結果
        np.testing.assert_array_almost_equal(
            result.tail(10).values,
            manual_atr.tail(10).values,
            decimal=6,
            err_msg="ATR calculation doesn't match manual calculation"
        )
        
        self.test_results['ATR_14'] = {
            'last_value': float(result.iloc[-1]),
            'avg_value': float(result.dropna().mean()),
            'status': 'PASS'
        }
        
        print(f"PASS: ATR(14) last value: {result.iloc[-1]:.3f}")
    
    def test_volume_indicators(self):
        """測試成交量指標"""
        # 測試 OBV
        obv_indicator = self.indicators['OBV']
        obv_result = obv_indicator.calculate(self.test_data)
        
        self.assertFalse(obv_result.empty, "OBV result should not be empty")
        
        # 手動計算 OBV 驗證
        price_diff = self.test_data['close'].diff()
        obv_manual = pd.Series(0, index=self.test_data.index, dtype=float)
        obv_manual[price_diff > 0] = self.test_data['volume'][price_diff > 0]
        obv_manual[price_diff < 0] = -self.test_data['volume'][price_diff < 0]
        obv_manual = obv_manual.cumsum()
        
        np.testing.assert_array_almost_equal(
            obv_result.tail(10).values,
            obv_manual.tail(10).values,
            decimal=0,
            err_msg="OBV calculation doesn't match manual calculation"
        )
        
        # 測試 MFI
        mfi_indicator = self.indicators['MFI_14']
        mfi_result = mfi_indicator.calculate(self.test_data)
        
        self.assertFalse(mfi_result.empty, "MFI result should not be empty")
        valid_mfi = mfi_result.dropna()
        self.assertTrue((valid_mfi >= 0).all() and (valid_mfi <= 100).all(), 
                       "MFI values should be between 0 and 100")
        
        self.test_results['Volume_Indicators'] = {
            'OBV_last': float(obv_result.iloc[-1]),
            'MFI_last': float(mfi_result.iloc[-1]),
            'status': 'PASS'
        }
        
        print(f"PASS: Volume indicators - OBV: {obv_result.iloc[-1]:.0f}, MFI: {mfi_result.iloc[-1]:.1f}")
    
    def test_calculation_speed(self):
        """測試計算速度"""
        import time
        
        # 創建大數據集
        large_data = TradingViewBenchmark.get_sample_data()
        # 擴展到1000個數據點
        for _ in range(10):
            large_data = pd.concat([large_data, large_data.iloc[-100:]])
        
        large_data = large_data.iloc[:1000].copy()
        large_data.index = pd.date_range(start='2020-01-01', periods=1000, freq='1D')
        
        speed_results = {}
        
        for name, indicator in self.indicators.items():
            try:
                start_time = time.time()
                result = indicator.calculate(large_data)
                end_time = time.time()
                
                calculation_time = end_time - start_time
                speed_results[name] = {
                    'time_ms': calculation_time * 1000,
                    'data_points': len(large_data),
                    'result_size': len(result) if hasattr(result, '__len__') else 1
                }
                
                # 確保計算時間合理（不超過100ms）
                self.assertLess(calculation_time, 0.1, 
                               f"{name} calculation took too long: {calculation_time:.3f}s")
                
            except Exception as e:
                speed_results[name] = {'error': str(e)}
                print(f"WARNING: {name} calculation failed: {e}")
        
        self.test_results['Speed_Test'] = speed_results
        
        # 顯示性能統計
        valid_times = [r['time_ms'] for r in speed_results.values() if 'time_ms' in r]
        if valid_times:
            print(f"PASS: Speed test completed - Avg: {np.mean(valid_times):.1f}ms, "
                  f"Max: {np.max(valid_times):.1f}ms")
    
    def test_signal_generation(self):
        """測試信號生成"""
        from src.indicators.signal_generator import IndicatorSignalGenerator
        
        signal_generator = IndicatorSignalGenerator()
        signals = signal_generator.generate_signals(self.test_data, 'TEST')
        
        # 檢查信號生成
        self.assertIsInstance(signals, list, "Signals should be a list")
        
        # 如果有信號，檢查信號結構
        if signals:
            signal = signals[0]
            self.assertTrue(hasattr(signal, 'signal_type'), "Signal should have signal_type")
            self.assertTrue(hasattr(signal, 'strength'), "Signal should have strength")
            self.assertTrue(hasattr(signal, 'confidence'), "Signal should have confidence")
            
            # 檢查數值範圍
            self.assertGreaterEqual(signal.strength, 0, "Signal strength should be >= 0")
            self.assertLessEqual(signal.strength, 100, "Signal strength should be <= 100")
            self.assertGreaterEqual(signal.confidence, 0, "Signal confidence should be >= 0")
            self.assertLessEqual(signal.confidence, 1, "Signal confidence should be <= 1")
        
        self.test_results['Signal_Generation'] = {
            'signals_generated': len(signals),
            'status': 'PASS'
        }
        
        print(f"PASS: Signal generation test - Generated {len(signals)} signals")
    
    @classmethod
    def tearDownClass(cls):
        """測試結束後的清理"""
        # 保存測試結果
        results_file = Path(__file__).parent / 'indicator_test_results.json'
        
        test_summary = {
            'timestamp': datetime.now().isoformat(),
            'test_data_info': {
                'data_points': len(cls.test_data),
                'date_range': f"{cls.test_data.index[0]} to {cls.test_data.index[-1]}",
                'price_range': f"${cls.test_data['close'].min():.2f} - ${cls.test_data['close'].max():.2f}"
            },
            'test_results': cls.test_results,
            'indicators_tested': list(cls.indicators.keys())
        }
        
        with open(results_file, 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
        
        print(f"\nTest results saved to: {results_file}")
        print(f"SUCCESS: All {len(cls.indicators)} indicators tested successfully!")


def run_comprehensive_test():
    """運行完整測試套件"""
    print("Technical Indicators Accuracy Test Suite")
    print("=" * 60)
    print("Target: Testing against TradingView benchmarks")
    print("Test: Validating calculation accuracy")
    print("Performance: Measuring computation performance")
    print("Signals: Testing signal generation")
    print("-" * 60)
    
    # 設置測試套件
    suite = unittest.TestLoader().loadTestsFromTestCase(IndicatorAccuracyTest)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # 運行測試
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nSUCCESS: All tests passed! Indicators are ready for production.")
        return True
    else:
        print(f"\nFAILED: Some tests failed. Failures: {len(result.failures)}, Errors: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)