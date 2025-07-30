# test_capital_quick.py
# 快速測試Capital.com API基本功能

import os
import sys
import logging
from datetime import datetime, timedelta

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 關閉config的調試輸出
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.stdout = open(os.devnull, 'w')
import config
sys.stdout = sys.__stdout__

# 導入數據模組
from data_pipeline import CapitalHistoryLoader, DataManager

def quick_test():
    """快速測試主要功能"""
    print("\n=== Capital.com API 快速測試 ===\n")
    
    try:
        # 1. 測試API連接
        print("1. 測試API連接...")
        loader = CapitalHistoryLoader()
        print("✅ API連接成功\n")
        
        # 2. 獲取AAPL最新價格
        print("2. 獲取AAPL最新價格...")
        market_info = loader.get_market_info("AAPL.US")
        if market_info and 'snapshot' in market_info:
            snapshot = market_info['snapshot']
            if snapshot.get('offer') and snapshot.get('bid'):
                price = (snapshot['offer'] + snapshot['bid']) / 2
                print(f"✅ AAPL.US 當前價格: ${price:.2f}\n")
        
        # 3. 獲取最近5天的日線數據
        print("3. 獲取AAPL最近5天的日線數據...")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        df = loader.get_bars("AAPL.US", "DAY", start_date, end_date)
        if not df.empty:
            print(f"✅ 獲取到 {len(df)} 筆數據")
            print("\n最新數據:")
            print(df.tail(3).to_string())
        
        loader.close()
        
        # 4. 測試數據管理器
        print("\n4. 測試數據管理器...")
        dm = DataManager(use_cache=True)
        
        # 獲取多個股票的最新價格
        symbols = ["AAPL.US", "MSFT.US", "GOOGL.US"]
        prices = dm.get_batch_latest_prices(symbols)
        
        print("\n最新價格:")
        for symbol, price in prices.items():
            print(f"  {symbol}: ${price:.2f}")
        
        dm.close()
        
        print("\n✅ 所有測試通過!")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()