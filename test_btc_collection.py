#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BTC Data Collection Test
測試加密貨幣數據收集（24/7市場）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import pandas as pd

class BTCDataCollector:
    def __init__(self):
        self.dwx = None
        self.tick_data = []
        self.crypto_symbols = []
        
    def connect(self):
        """連接到MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='BTCCollector',
            _host='localhost',
            _protocol='tcp',
            _PUSH_PORT=32768,
            _PULL_PORT=32769,
            _SUB_PORT=32770,
            _verbose=False,
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
                    account_info = info_list[0]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to Account: {acc_num}")
                    print(f"  Balance: ${account_info.get('account_balance', 0):.2f}")
                    print(f"  Leverage: 1:{account_info.get('account_leverage', 100)}")
                    return True
        
        print("[INFO] Connected to MT4")
        return True
    
    def find_crypto_symbols(self):
        """查找可用的加密貨幣交易對"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Searching for crypto symbols...")
        
        # 常見的加密貨幣符號變體
        crypto_variants = [
            'BTCUSD', 'Bitcoin', 'BTC/USD', 'BTCUSD.', 'BTCUSDm',
            'ETHUSD', 'Ethereum', 'ETH/USD', 'ETHUSD.', 'ETHUSDm',
            'XRPUSD', 'Ripple', 'XRP/USD', 'XRPUSD.', 'XRPUSDm',
            'LTCUSD', 'Litecoin', 'LTC/USD', 'LTCUSD.', 'LTCUSDm',
            'BCHUSD', 'BitcoinCash', 'BCH/USD', 'BCHUSD.', 'BCHUSDm'
        ]
        
        # 測試每個符號
        available_symbols = []
        for symbol in crypto_variants:
            print(f"  Testing {symbol}...", end='')
            
            # 嘗試訂閱
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(0.5)
            
            # 檢查是否有數據
            if symbol in self.dwx._Market_Data_DB:
                print(" [FOUND]")
                available_symbols.append(symbol)
            else:
                print(" [NOT FOUND]")
                # 取消訂閱失敗的符號
                try:
                    self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
                except:
                    pass
        
        self.crypto_symbols = available_symbols
        
        if available_symbols:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(available_symbols)} crypto symbols:")
            for sym in available_symbols:
                print(f"  - {sym}")
        else:
            print(f"\n[WARNING] No crypto symbols found. Will try default forex pairs.")
            # 如果沒有找到加密貨幣，使用外匯對
            self.crypto_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            for sym in self.crypto_symbols:
                self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(sym)
        
        return available_symbols
    
    def collect_data(self, duration_seconds=30):
        """收集數據"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting data for {duration_seconds} seconds...")
        print("="*60)
        
        start_time = time.time()
        last_display_time = start_time
        tick_count = 0
        last_prices = {}
        
        while time.time() - start_time < duration_seconds:
            # 檢查市場數據
            if self.dwx._Market_Data_DB:
                for symbol, data in self.dwx._Market_Data_DB.items():
                    if data and symbol in self.crypto_symbols:
                        # 獲取最新的tick
                        latest_timestamp = list(data.keys())[-1]
                        bid, ask = data[latest_timestamp]
                        
                        # 計算價格變化
                        price_change = ""
                        if symbol in last_prices:
                            last_bid = last_prices[symbol]
                            if bid > last_bid:
                                price_change = "↑"
                            elif bid < last_bid:
                                price_change = "↓"
                            else:
                                price_change = "="
                        
                        last_prices[symbol] = bid
                        
                        # 存儲tick數據
                        self.tick_data.append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'bid': bid,
                            'ask': ask,
                            'spread': round((ask - bid) * 10000, 2) if 'USD' in symbol else round(ask - bid, 2)
                        })
                        tick_count += 1
                        
                        # 每秒顯示一次最新價格
                        current_time = time.time()
                        if current_time - last_display_time >= 1:
                            # 根據符號類型格式化價格
                            if 'BTC' in symbol:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: ${bid:,.2f} / ${ask:,.2f} {price_change} Spread: ${ask-bid:.2f}")
                            else:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: {bid:.5f} / {ask:.5f} {price_change} Spread: {round((ask - bid) * 10000, 2)} pts")
                            last_display_time = current_time
            
            time.sleep(0.1)
        
        print("="*60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection complete. Total ticks: {tick_count}")
        return tick_count
    
    def show_statistics(self):
        """顯示統計信息"""
        if not self.tick_data:
            print("[WARNING] No data collected")
            return
        
        df = pd.DataFrame(self.tick_data)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Data Statistics:")
        print("="*60)
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            print(f"\n{symbol}:")
            print(f"  Ticks collected: {len(symbol_data)}")
            print(f"  Bid range: {symbol_data['bid'].min():.5f} - {symbol_data['bid'].max():.5f}")
            print(f"  Average spread: {symbol_data['spread'].mean():.2f}")
            print(f"  Max spread: {symbol_data['spread'].max():.2f}")
            print(f"  Min spread: {symbol_data['spread'].min():.2f}")
    
    def save_data(self, filename='btc_data.csv'):
        """保存數據"""
        if self.tick_data:
            df = pd.DataFrame(self.tick_data)
            df.to_csv(filename, index=False)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Data saved to {filename}")
            return df
        return None
    
    def disconnect(self):
        """斷開連接"""
        if self.dwx:
            # 取消所有訂閱
            for symbol in self.crypto_symbols:
                try:
                    self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
                except:
                    pass
            
            # 關閉連接
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnected from MT4")

def main():
    """主函數"""
    print("\n" + "="*60)
    print(" BTC/Crypto Real-time Data Collection ")
    print("="*60)
    print(f" Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Crypto markets are open 24/7")
    print("="*60)
    
    # 創建收集器
    collector = BTCDataCollector()
    
    try:
        # 連接
        if collector.connect():
            # 查找加密貨幣符號
            crypto_symbols = collector.find_crypto_symbols()
            
            if not crypto_symbols:
                print("\n[INFO] Will collect available forex pairs instead")
            
            # 收集數據30秒
            tick_count = collector.collect_data(duration_seconds=30)
            
            # 顯示統計
            collector.show_statistics()
            
            # 保存數據
            if tick_count > 0:
                collector.save_data('crypto_ticks.csv')
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 斷開連接
        collector.disconnect()
    
    print("\n" + "="*60)
    print(" Test Complete ")
    print("="*60)

if __name__ == "__main__":
    main()