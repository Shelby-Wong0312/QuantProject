#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BTC Real-time Data Collection via MT4
Symbol: BTCUSD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import pandas as pd
import json

class BTCCollector:
    def __init__(self):
        self.dwx = None
        self.btc_data = []
        self.last_price = None
        
    def connect(self):
        """連接到MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        # 創建DWX連接器
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='BTCCollector',
            _host='localhost',
            _protocol='tcp',
            _PUSH_PORT=32768,
            _PULL_PORT=32769,
            _SUB_PORT=32770,
            _verbose=False,  # 設為False減少輸出
            _poll_timeout=1000,
            _sleep_delay=0.001
        )
        
        # 等待連接建立
        time.sleep(2)
        
        # 獲取帳戶信息
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Getting account info...")
        self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        # 顯示帳戶信息
        if hasattr(self.dwx, 'account_info_DB') and self.dwx.account_info_DB:
            for acc_num, info_list in self.dwx.account_info_DB.items():
                if info_list:
                    account_info = info_list[0]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Account: {acc_num}")
                    print(f"  Balance: ${account_info.get('account_balance', 0):.2f}")
                    print(f"  Equity: ${account_info.get('account_equity', 0):.2f}")
                    print(f"  Leverage: 1:{account_info.get('account_leverage', 100)}")
        
        return True
    
    def subscribe_btc(self):
        """訂閱BTCUSD"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Subscribing to BTCUSD...")
        self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('BTCUSD')
        time.sleep(1)
        
        # 也訂閱其他主要符號進行比較
        other_symbols = ['EURUSD', 'XAUUSD']  # EUR/USD和黃金
        for symbol in other_symbols:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Also subscribing to {symbol}...")
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(0.5)
    
    def collect_data(self, duration_seconds=60):
        """收集數據"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting data collection for {duration_seconds} seconds...")
        print("="*60)
        
        start_time = time.time()
        last_display_time = start_time
        tick_count = 0
        
        while time.time() - start_time < duration_seconds:
            # 檢查市場數據
            if self.dwx._Market_Data_DB:
                current_time = time.time()
                
                for symbol, data in self.dwx._Market_Data_DB.items():
                    if data:
                        # 獲取最新的tick
                        timestamps = list(data.keys())
                        if timestamps:
                            latest_timestamp = timestamps[-1]
                            bid, ask = data[latest_timestamp]
                            
                            # 保存BTC數據
                            if symbol == 'BTCUSD':
                                tick_data = {
                                    'timestamp': datetime.now(),
                                    'bid': bid,
                                    'ask': ask,
                                    'spread': ask - bid,
                                    'mid': (bid + ask) / 2
                                }
                                self.btc_data.append(tick_data)
                                tick_count += 1
                                
                                # 計算價格變化
                                price_change = ""
                                change_amount = 0
                                if self.last_price:
                                    change_amount = bid - self.last_price
                                    if change_amount > 0:
                                        price_change = f"↑ +${change_amount:.2f}"
                                    elif change_amount < 0:
                                        price_change = f"↓ ${change_amount:.2f}"
                                    else:
                                        price_change = "="
                                
                                self.last_price = bid
                                
                                # 每秒顯示一次
                                if current_time - last_display_time >= 1:
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] BTC/USD: ${bid:,.2f} / ${ask:,.2f} | Spread: ${ask-bid:.2f} {price_change}")
                                    last_display_time = current_time
                            
                            # 顯示其他符號
                            elif current_time - last_display_time >= 1:
                                if symbol == 'EURUSD':
                                    print(f"           EUR/USD: {bid:.5f} / {ask:.5f}")
                                elif symbol == 'XAUUSD':
                                    print(f"           Gold:    ${bid:.2f} / ${ask:.2f}")
            
            time.sleep(0.1)  # 100ms
            
            # 定期顯示統計
            elapsed = time.time() - start_time
            if elapsed > 0 and int(elapsed) % 10 == 0 and elapsed - int(elapsed) < 0.1:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: {int(elapsed)}/{duration_seconds}s | Ticks collected: {tick_count}")
                if self.btc_data:
                    prices = [d['bid'] for d in self.btc_data]
                    print(f"  BTC Stats - Min: ${min(prices):,.2f} | Max: ${max(prices):,.2f} | Current: ${prices[-1]:,.2f}")
                print()
        
        print("="*60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection complete. Total BTC ticks: {tick_count}")
        return tick_count
    
    def analyze_data(self):
        """分析收集的數據"""
        if not self.btc_data:
            print("\n[WARNING] No BTC data collected")
            return None
        
        df = pd.DataFrame(self.btc_data)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] BTC/USD Analysis:")
        print("="*60)
        
        print(f"Total ticks: {len(df)}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nPrice Statistics:")
        print(f"  Bid  - Min: ${df['bid'].min():,.2f} | Max: ${df['bid'].max():,.2f} | Mean: ${df['bid'].mean():,.2f}")
        print(f"  Ask  - Min: ${df['ask'].min():,.2f} | Max: ${df['ask'].max():,.2f} | Mean: ${df['ask'].mean():,.2f}")
        print(f"  Spread - Min: ${df['spread'].min():.2f} | Max: ${df['spread'].max():.2f} | Mean: ${df['spread'].mean():.2f}")
        
        # 計算波動性
        if len(df) > 1:
            returns = df['mid'].pct_change().dropna()
            volatility = returns.std() * 100
            print(f"  Volatility: {volatility:.4f}%")
        
        return df
    
    def save_data(self, filename='btc_data.csv'):
        """保存數據到CSV"""
        if self.btc_data:
            df = pd.DataFrame(self.btc_data)
            df.to_csv(filename, index=False)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Data saved to {filename}")
            return df
        else:
            print("\n[WARNING] No data to save")
            return None
    
    def disconnect(self):
        """斷開連接"""
        if self.dwx:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            
            # 取消訂閱
            for symbol in ['BTCUSD', 'EURUSD', 'XAUUSD']:
                try:
                    self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
                except:
                    pass
            
            # 關閉連接
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Disconnected from MT4")

def main():
    """主函數"""
    print("\n" + "="*70)
    print(" BTC Real-time Data Collection via MT4 ")
    print("="*70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Symbol: BTCUSD")
    print(f" Note: Crypto markets are open 24/7")
    print("="*70)
    
    collector = BTCCollector()
    
    try:
        # 連接到MT4
        if collector.connect():
            # 訂閱BTCUSD
            collector.subscribe_btc()
            
            # 收集數據30秒
            print("\nPress Ctrl+C to stop early")
            tick_count = collector.collect_data(duration_seconds=30)
            
            # 分析數據
            if tick_count > 0:
                df = collector.analyze_data()
                
                # 保存數據
                collector.save_data('btc_realtime.csv')
                
                # 顯示最後5個tick
                if df is not None and len(df) > 0:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Last 5 ticks:")
                    print(df[['timestamp', 'bid', 'ask', 'spread']].tail(5).to_string(index=False))
            else:
                print("\n[WARNING] No data was collected. Please check:")
                print("1. Is BTCUSD available in your MT4?")
                print("2. Is the market open?")
                print("3. Try other symbols like EURUSD")
    
    except KeyboardInterrupt:
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 確保斷開連接
        collector.disconnect()
    
    print("\n" + "="*70)
    print(" Collection Complete ")
    print("="*70)

if __name__ == "__main__":
    main()