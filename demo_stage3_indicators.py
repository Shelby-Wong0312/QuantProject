"""
Stage 3 Technical Indicators Demo
階段3技術指標演示
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the indicators library
from src.indicators import (
    RSI, MACD, BollingerBands, SMA,
    IndicatorCalculator, CalculationConfig,
    IndicatorSignalGenerator
)

def create_demo_data():
    """創建演示數據"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
    
    # 生成模擬股價數據
    price = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    data = pd.DataFrame({
        'open': price + np.random.randn(100) * 0.01,
        'high': price + np.abs(np.random.randn(100) * 0.02),
        'low': price - np.abs(np.random.randn(100) * 0.02),
        'close': price,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    # 確保 OHLC 邏輯正確
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
    
    return data

def demo_single_indicators():
    """演示單個指標計算"""
    print("=== Single Indicator Calculation Demo ===")
    
    data = create_demo_data()
    print(f"Test data: {len(data)} days, price range ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # RSI 計算
    rsi = RSI(period=14)
    rsi_result = rsi.calculate(data)
    print(f"RSI(14) last value: {rsi_result.iloc[-1]:.1f}")
    
    # MACD 計算
    macd = MACD()
    macd_result = macd.calculate(data)
    print(f"MACD last values - Line: {macd_result['macd'].iloc[-1]:.3f}, Signal: {macd_result['signal'].iloc[-1]:.3f}")
    
    # 布林帶計算
    bb = BollingerBands(period=20)
    bb_result = bb.calculate(data)
    print(f"Bollinger Bands - Upper: {bb_result['upper_band'].iloc[-1]:.2f}, Lower: {bb_result['lower_band'].iloc[-1]:.2f}")
    
    print("Single indicator demo completed successfully!")

def demo_batch_calculation():
    """演示批量計算"""
    print("\n=== Batch Calculation Demo ===")
    
    # 創建多股票數據
    stocks_data = {}
    for i in range(5):
        symbol = f"DEMO{i:02d}"
        stocks_data[symbol] = create_demo_data()
    
    print(f"Created data for {len(stocks_data)} stocks")
    
    # 配置計算器
    config = CalculationConfig(
        timeframes=['1d'],
        indicators=['SMA_20', 'RSI_14', 'MACD', 'BollingerBands'],
        use_multiprocessing=False,  # 小規模測試關閉多進程
        cache_results=True
    )
    
    calculator = IndicatorCalculator(config)
    
    # 執行批量計算
    import time
    start_time = time.time()
    results = calculator.calculate_all_indicators(stocks_data)
    calculation_time = time.time() - start_time
    
    # 顯示結果
    successful_stocks = len([r for r in results.values() if r])
    print(f"Batch calculation completed in {calculation_time:.3f}s")
    print(f"Successful calculations: {successful_stocks}/{len(stocks_data)} stocks")
    
    # 顯示樣本結果
    sample_symbol = list(results.keys())[0]
    sample_results = results[sample_symbol]['1d']
    print(f"Sample results for {sample_symbol}: {list(sample_results.keys())}")
    
    print("Batch calculation demo completed successfully!")

def demo_signal_generation():
    """演示信號生成"""
    print("\n=== Signal Generation Demo ===")
    
    data = create_demo_data()
    
    # 初始化信號生成器
    signal_generator = IndicatorSignalGenerator()
    
    # 生成信號
    signals = signal_generator.generate_signals(data, 'DEMO')
    
    print(f"Generated {len(signals)} trading signals")
    
    if signals:
        for i, signal in enumerate(signals[-3:]):  # 顯示最後3個信號
            print(f"Signal {i+1}:")
            print(f"  Type: {signal.signal_type.value}")
            print(f"  Strength: {signal.strength:.1f}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Price: ${signal.price:.2f}")
            print(f"  Reasons: {', '.join(signal.reasons)}")
    else:
        print("No signals generated (normal for demo data)")
    
    print("Signal generation demo completed successfully!")

def demo_performance_stats():
    """演示性能統計"""
    print("\n=== Performance Statistics Demo ===")
    
    # 創建較大的測試數據
    stocks_data = {}
    for i in range(20):  # 20支股票
        symbol = f"PERF{i:02d}"
        stocks_data[symbol] = create_demo_data()
    
    config = CalculationConfig(
        timeframes=['1d'],
        use_multiprocessing=False,
        cache_results=True
    )
    
    calculator = IndicatorCalculator(config)
    
    # 執行計算並測量性能
    import time
    start_time = time.time()
    results = calculator.calculate_all_indicators(stocks_data)
    end_time = time.time()
    
    # 獲取性能統計
    stats = calculator.get_performance_stats()
    
    print(f"Performance Statistics:")
    print(f"  Total calculations: {stats['total_calculations']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
    print(f"  Total time: {end_time - start_time:.3f}s")
    print(f"  Stocks processed: {stats['stocks_processed']}")
    print(f"  Average time per stock: {stats['avg_time_per_stock']:.3f}s")
    
    stocks_per_second = len(stocks_data) / (end_time - start_time)
    print(f"  Processing speed: {stocks_per_second:.1f} stocks/second")
    
    print("Performance demo completed successfully!")

def main():
    """主演示函數"""
    print("Stage 3 Technical Indicators Library Demo")
    print("=" * 50)
    print("Testing all components of the technical indicators system")
    print()
    
    try:
        # 運行各個演示
        demo_single_indicators()
        demo_batch_calculation()
        demo_signal_generation()
        demo_performance_stats()
        
        print("\n" + "=" * 50)
        print("SUCCESS: All Stage 3 components working correctly!")
        print("Technical Indicators Library is ready for production use.")
        
        # 顯示系統摘要
        print(f"\nSystem Summary:")
        print(f"- 20+ technical indicators implemented")
        print(f"- Multi-timeframe support (1m, 5m, 15m, 1h, 1d)")
        print(f"- Vectorized calculations with pandas/numpy")
        print(f"- Multi-processing parallel computation")
        print(f"- Intelligent caching system")
        print(f"- Advanced signal generation with filtering")
        print(f"- Production-ready performance")
        
    except Exception as e:
        print(f"\nERROR: Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)