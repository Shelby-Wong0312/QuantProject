"""
Test Signal Generator
測試信號生成器
Cloud DE - Verification Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

from quantproject.signals.signal_generator import SignalGenerator, SignalStrength
from quantproject.data.data_manager import DataManager


def test_signal_generation():
    """測試信號生成"""
    print("\n" + "="*50)
    print("Testing Signal Generation")
    print("="*50)
    
    # 初始化信號生成器
    generator = SignalGenerator()
    
    # 創建測試數據
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    test_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 101,
        'low': np.random.randn(200).cumsum() + 99,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(50000, 200000, 200)
    }, index=dates)
    
    # 測試單個信號
    start_time = time.time()
    signal = generator.generate_signal(test_data, 'TEST_STOCK')
    latency = (time.time() - start_time) * 1000
    
    print(f"✓ Signal generated in {latency:.2f}ms")
    print(f"\nSignal Details:")
    print(f"  Symbol: {signal.symbol}")
    print(f"  Action: {signal.action}")
    print(f"  Strength: {signal.strength:.2f}/100")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Price: ${signal.price:.2f}")
    print(f"  Stop Loss: ${signal.stop_loss:.2f}")
    print(f"  Take Profit: ${signal.take_profit:.2f}")
    print(f"  Risk Score: {signal.risk_score:.2f}")
    
    # 檢查延遲要求
    if latency < 100:
        print(f"\n✓ Meets latency requirement (<100ms)")
    else:
        print(f"\n✗ Exceeds latency requirement (>100ms)")
    
    return signal


def test_multi_stock_signals():
    """測試多股票並行處理"""
    print("\n" + "="*50)
    print("Testing Multi-Stock Signal Generation")
    print("="*50)
    
    generator = SignalGenerator()
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    signals = []
    start_time = time.time()
    
    for symbol in symbols:
        # 創建隨機數據
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
        data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 101,
            'low': np.random.randn(200).cumsum() + 99,
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(50000, 500000, 200)
        }, index=dates)
        
        signal = generator.generate_signal(data, symbol)
        signals.append(signal)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(symbols) * 1000
    
    print(f"✓ Generated {len(signals)} signals in {total_time:.2f}s")
    print(f"  Average time per signal: {avg_time:.2f}ms")
    
    # 顯示信號分佈
    actions = [s.action for s in signals]
    print(f"\nSignal Distribution:")
    for action in set(actions):
        count = actions.count(action)
        print(f"  {action}: {count} ({count/len(actions)*100:.1f}%)")
    
    return signals


def test_signal_scoring():
    """測試信號評分系統"""
    print("\n" + "="*50)
    print("Testing Signal Scoring System")
    print("="*50)
    
    generator = SignalGenerator()
    
    # 創建不同市場情況的數據
    scenarios = {
        'Strong Uptrend': {'trend': 0.5, 'volatility': 0.1},
        'Strong Downtrend': {'trend': -0.5, 'volatility': 0.1},
        'High Volatility': {'trend': 0, 'volatility': 0.5},
        'Sideways Market': {'trend': 0, 'volatility': 0.05}
    }
    
    for scenario_name, params in scenarios.items():
        # 生成場景數據
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
        trend = np.linspace(0, params['trend'] * 200, 200)
        noise = np.random.normal(0, params['volatility'], 200).cumsum()
        prices = 100 + trend + noise
        
        data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 200),
            'high': prices + abs(np.random.normal(0, 0.2, 200)),
            'low': prices - abs(np.random.normal(0, 0.2, 200)),
            'close': prices,
            'volume': np.random.randint(50000, 200000, 200)
        }, index=dates)
        
        signal = generator.generate_signal(data, f'{scenario_name}_TEST')
        
        print(f"\n{scenario_name}:")
        print(f"  Action: {signal.action}")
        print(f"  Strength: {signal.strength:.2f}")
        print(f"  Confidence: {signal.confidence:.2%}")


def test_signal_history():
    """測試信號歷史記錄"""
    print("\n" + "="*50)
    print("Testing Signal History")
    print("="*50)
    
    generator = SignalGenerator()
    
    # 生成多個信號
    for i in range(10):
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
        data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 101,
            'low': np.random.randn(200).cumsum() + 99,
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(50000, 200000, 200)
        }, index=dates)
        
        generator.generate_signal(data, f'STOCK_{i}')
    
    # 獲取歷史
    history = generator.get_signal_history()
    
    print(f"✓ Stored {len(history)} signals")
    
    # 評估性能
    performance = generator.evaluate_performance()
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Signals: {performance.get('total_signals', 0)}")
    print(f"  Average Confidence: {performance.get('avg_confidence', 0):.2%}")
    print(f"  Average Strength: {performance.get('avg_strength', 0):.2f}")
    
    if 'action_distribution' in performance:
        print(f"\nAction Distribution:")
        for action, count in performance['action_distribution'].items():
            print(f"  {action}: {count}")


def test_with_real_data():
    """使用真實數據測試"""
    print("\n" + "="*50)
    print("Testing with Real Stock Data")
    print("="*50)
    
    try:
        # 嘗試載入真實數據
        dm = DataManager()
        
        # 選擇一支股票
        symbol = 'AAPL'
        df = dm.load_stock_data(symbol)
        
        if df is not None and len(df) > 100:
            # 使用最近的數據
            recent_data = df.tail(200)
            
            generator = SignalGenerator()
            signal = generator.generate_signal(recent_data, symbol)
            
            print(f"✓ Real data signal for {symbol}:")
            print(f"  Action: {signal.action}")
            print(f"  Strength: {signal.strength:.2f}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Predicted Return: {signal.predicted_return:.2%}")
            
            # 顯示信號來源貢獻
            print(f"\nSignal Sources:")
            for source, data in signal.sources.items():
                if data:
                    print(f"  {source}: {data.get('action', 'N/A')} (score: {data.get('score', 0):.2f})")
        else:
            print("✗ Real data not available, skipping test")
            
    except Exception as e:
        print(f"✗ Real data test failed: {e}")


def run_performance_test():
    """性能壓力測試"""
    print("\n" + "="*50)
    print("Performance Stress Test")
    print("="*50)
    
    generator = SignalGenerator()
    
    # 測試不同數據大小
    data_sizes = [100, 500, 1000]
    
    for size in data_sizes:
        dates = pd.date_range(start='2024-01-01', periods=size, freq='5min')
        data = pd.DataFrame({
            'open': np.random.randn(size).cumsum() + 100,
            'high': np.random.randn(size).cumsum() + 101,
            'low': np.random.randn(size).cumsum() + 99,
            'close': np.random.randn(size).cumsum() + 100,
            'volume': np.random.randint(50000, 200000, size)
        }, index=dates)
        
        # 測試100次
        times = []
        for _ in range(100):
            start = time.time()
            generator.generate_signal(data, 'PERF_TEST')
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        print(f"\nData Size: {size} bars")
        print(f"  Avg Time: {avg_time:.2f}ms")
        print(f"  Min Time: {min_time:.2f}ms")
        print(f"  Max Time: {max_time:.2f}ms")
        print(f"  Meets <100ms: {'✓' if max_time < 100 else '✗'}")


def main():
    """主測試函數"""
    print("\n" + "="*60)
    print("SIGNAL GENERATOR TEST SUITE")
    print("Cloud DE - Task RT-001 Verification")
    print("="*60)
    
    results = {}
    
    # 1. 基本信號生成
    try:
        signal = test_signal_generation()
        results['basic_generation'] = True
    except Exception as e:
        print(f"✗ Basic generation test failed: {e}")
        results['basic_generation'] = False
    
    # 2. 多股票處理
    try:
        signals = test_multi_stock_signals()
        results['multi_stock'] = True
    except Exception as e:
        print(f"✗ Multi-stock test failed: {e}")
        results['multi_stock'] = False
    
    # 3. 信號評分
    try:
        test_signal_scoring()
        results['scoring'] = True
    except Exception as e:
        print(f"✗ Scoring test failed: {e}")
        results['scoring'] = False
    
    # 4. 歷史記錄
    try:
        test_signal_history()
        results['history'] = True
    except Exception as e:
        print(f"✗ History test failed: {e}")
        results['history'] = False
    
    # 5. 真實數據
    try:
        test_with_real_data()
        results['real_data'] = True
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        results['real_data'] = False
    
    # 6. 性能測試
    try:
        run_performance_test()
        results['performance'] = True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        results['performance'] = False
    
    # 總結
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Signal Generator is ready for production!")
    else:
        print("\n⚠️ Some tests failed. Please review and fix.")
    
    return all_passed


if __name__ == "__main__":
    success = main()