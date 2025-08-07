#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4 Real-time Data Collection Script
使用DWX連接器收集MT4數據
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import pandas as pd

class MT4DataCollector:
    def __init__(self):
        self.dwx = None
        self.tick_data = []
        self.account_info = {}
        
    def connect(self):
        """連接到MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='DataCollector',
            _host='localhost',
            _protocol='tcp',
            _PUSH_PORT=32768,
            _PULL_PORT=32769,
            _SUB_PORT=32770,
            _verbose=False,  # 減少輸出
            _poll_timeout=1000,
            _sleep_delay=0.001
        )
        
        time.sleep(2)
        
        # 獲取帳戶信息
        self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'account_info_DB') and self.dwx.account_info_DB:
            for acc_num, info_list in self.dwx.account_info_DB.items():
                if info_list:
                    self.account_info = info_list[0]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to Account: {acc_num}")
                    print(f"  Balance: ${self.account_info.get('account_balance', 0):.2f}")
                    print(f"  Equity: ${self.account_info.get('account_equity', 0):.2f}")
                    return True
        
        print("[WARNING] Connected but no account info received")
        return True
    
    def subscribe_symbols(self, symbols):
        """訂閱交易品種"""
        for symbol in symbols:
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Subscribed to {symbol}")
            time.sleep(0.5)
    
    def collect_data(self, duration_seconds=60):
        """收集數據指定時間"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting data for {duration_seconds} seconds...")
        
        start_time = time.time()
        last_display_time = start_time
        tick_count = 0
        
        while time.time() - start_time < duration_seconds:
            # 檢查市場數據
            if self.dwx._Market_Data_DB:
                for symbol, data in self.dwx._Market_Data_DB.items():
                    if data:
                        # 獲取最新的tick
                        latest_timestamp = list(data.keys())[-1]
                        bid, ask = data[latest_timestamp]
                        
                        # 存儲tick數據
                        self.tick_data.append({
                            'timestamp': latest_timestamp,
                            'symbol': symbol,
                            'bid': bid,
                            'ask': ask,
                            'spread': round((ask - bid) * 10000, 2)  # 點差（點）
                        })
                        tick_count += 1
                        
                        # 每秒顯示一次最新價格
                        current_time = time.time()
                        if current_time - last_display_time >= 1:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: {bid:.5f}/{ask:.5f} Spread: {round((ask - bid) * 10000, 2)} pts")
                            last_display_time = current_time
            
            # 檢查開倉交易
            if tick_count % 100 == 0 and tick_count > 0:
                self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            
            time.sleep(0.1)  # 100ms
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collection complete. Total ticks: {tick_count}")
        return tick_count
    
    def save_data(self, filename='mt4_data.csv'):
        """保存數據到CSV"""
        if self.tick_data:
            df = pd.DataFrame(self.tick_data)
            df.to_csv(filename, index=False)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Data saved to {filename}")
            return df
        else:
            print("[WARNING] No data to save")
            return None
    
    def disconnect(self):
        """斷開連接"""
        if self.dwx:
            # 取消所有訂閱
            for symbol in list(self.dwx._Market_Data_DB.keys()):
                self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
            
            # 關閉連接
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnected from MT4")

def main():
    """主函數"""
    print("\n" + "="*60)
    print(" MT4 Real-time Data Collection ")
    print("="*60)
    
    # 檢查是否為交易時間
    now = datetime.now()
    if now.weekday() >= 5:
        print("\n[WARNING] It's weekend. Forex market is closed.")
        print("You may not receive real-time tick data.")
    
    # 創建收集器
    collector = MT4DataCollector()
    
    try:
        # 連接
        if collector.connect():
            # 訂閱品種
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            collector.subscribe_symbols(symbols)
            
            # 收集數據60秒
            tick_count = collector.collect_data(duration_seconds=60)
            
            # 保存數據
            if tick_count > 0:
                df = collector.save_data('mt4_ticks.csv')
                if df is not None:
                    print("\nData Summary:")
                    print(df.describe())
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # 斷開連接
        collector.disconnect()
    
    print("\n" + "="*60)
    print(" Collection Complete ")
    print("="*60)

if __name__ == "__main__":
    main()