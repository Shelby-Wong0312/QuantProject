# quant_project/test_capital_api.py
# 測試Capital.com API功能

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_capital_api():
    """測試Capital.com API的各項功能"""
    
    print("="*60)
    print("Capital.com API 功能測試")
    print("="*60)
    
    # 初始化數據管理器
    from data_pipeline.data_manager import DataManager
    data_manager = DataManager(use_cache=True)
    
    # 測試1: 獲取可用交易品種
    print("\n1. 測試獲取可用交易品種...")
    try:
        symbols = data_manager.get_available_symbols()
        print(f"✅ 成功獲取 {len(symbols)} 個可用交易品種")
        print(f"   前10個品種: {symbols[:10]}")
    except Exception as e:
        print(f"❌ 獲取交易品種失敗: {e}")
    
    # 測試2: 獲取單個股票的歷史數據
    print("\n2. 測試獲取Apple(AAPL)的歷史數據...")
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        df = data_manager.get_historical_data(
            symbol="AAPL.US",
            resolution="DAY",
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            print(f"✅ 成功獲取 {len(df)} 筆歷史數據")
            print("\n最近5筆數據:")
            print(df.tail())
            print(f"\n數據範圍: {df.index[0]} 到 {df.index[-1]}")
        else:
            print("❌ 未獲取到數據")
    except Exception as e:
        print(f"❌ 獲取歷史數據失敗: {e}")
    
    # 測試3: 獲取最新價格
    print("\n3. 測試獲取最新價格...")
    test_symbols = ["AAPL.US", "MSFT.US", "GOOGL.US"]
    for symbol in test_symbols:
        try:
            price = data_manager.get_latest_price(symbol)
            if price:
                print(f"✅ {symbol}: ${price:.2f}")
            else:
                print(f"❌ {symbol}: 無法獲取價格")
        except Exception as e:
            print(f"❌ {symbol}: 錯誤 - {e}")
    
    # 測試4: 批量獲取多個股票數據
    print("\n4. 測試批量獲取多個股票的歷史數據...")
    try:
        symbols_to_test = ["AAPL.US", "MSFT.US", "GOOGL.US", "TSLA.US"]
        results = data_manager.get_multiple_symbols_data(
            symbols=symbols_to_test,
            resolution="DAY",
            lookback_days=10
        )
        
        print(f"✅ 成功獲取 {len(results)} 個股票的數據:")
        for symbol, df in results.items():
            if not df.empty:
                print(f"   {symbol}: {len(df)} 筆數據")
            else:
                print(f"   {symbol}: 無數據")
    except Exception as e:
        print(f"❌ 批量獲取失敗: {e}")
    
    # 測試5: 測試緩存功能
    print("\n5. 測試緩存功能...")
    try:
        # 獲取緩存統計
        stats = data_manager.get_cache_stats()
        print(f"✅ 緩存統計:")
        print(f"   總文件數: {stats.get('total_files', 0)}")
        print(f"   總大小: {stats.get('total_size_mb', 0)} MB")
        print(f"   總記錄數: {stats.get('total_records', 0)}")
        print(f"   緩存的股票: {stats.get('symbols', [])}")
        
        # 再次獲取相同數據（應該從緩存讀取）
        print("\n   測試從緩存讀取數據...")
        import time
        start_time = time.time()
        df_cached = data_manager.get_historical_data(
            symbol="AAPL.US",
            resolution="DAY",
            start_date=start_date,
            end_date=end_date
        )
        elapsed_time = time.time() - start_time
        print(f"   從緩存讀取耗時: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 緩存測試失敗: {e}")
    
    # 測試6: 測試不同時間週期
    print("\n6. 測試不同時間週期的數據獲取...")
    resolutions = ["MINUTE", "MINUTE_5", "HOUR", "DAY"]
    for resolution in resolutions:
        try:
            df = data_manager.get_historical_data(
                symbol="AAPL.US",
                resolution=resolution,
                lookback_days=1
            )
            if not df.empty:
                print(f"✅ {resolution}: 獲取 {len(df)} 筆數據")
            else:
                print(f"❌ {resolution}: 無數據")
        except Exception as e:
            print(f"❌ {resolution}: 錯誤 - {e}")
    
    # 關閉連接
    data_manager.close()
    
    print("\n" + "="*60)
    print("測試完成！")
    print("="*60)

if __name__ == "__main__":
    test_capital_api()