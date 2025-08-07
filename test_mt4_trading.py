#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4 自動交易功能測試腳本
測試項目：
1. 開倉 (Buy/Sell)
2. 平倉 (Close)
3. 修改訂單 (Modify)
4. 查詢帳戶資訊
5. 查詢持倉
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import json

class MT4TradingTester:
    def __init__(self):
        self.dwx = None
        self.test_results = []
        
    def connect(self):
        """連接到MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='TradingTester',
            _verbose=True,
            _poll_timeout=1000
        )
        
        time.sleep(3)
        return True
    
    def test_account_info(self):
        """測試1: 獲取帳戶資訊"""
        print(f"\n{'='*60}")
        print(f"TEST 1: Account Information")
        print(f"{'='*60}")
        
        self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'account_info_DB') and self.dwx.account_info_DB:
            for acc_num, info_list in self.dwx.account_info_DB.items():
                if info_list:
                    info = info_list[0]
                    print(f"Account Number: {acc_num}")
                    print(f"Balance: ${info.get('account_balance', 0):.2f}")
                    print(f"Equity: ${info.get('account_equity', 0):.2f}")
                    print(f"Margin: ${info.get('account_margin', 0):.2f}")
                    print(f"Free Margin: ${info.get('account_freemargin', 0):.2f}")
                    print(f"Leverage: 1:{info.get('account_leverage', 100)}")
                    
                    self.test_results.append({
                        'test': 'Account Info',
                        'status': 'PASSED',
                        'balance': info.get('account_balance', 0)
                    })
                    return True
        
        print("[ERROR] Failed to get account info")
        self.test_results.append({'test': 'Account Info', 'status': 'FAILED'})
        return False
    
    def test_open_position(self, symbol='BTCUSD', volume=0.01):
        """測試2: 開倉測試"""
        print(f"\n{'='*60}")
        print(f"TEST 2: Open Position - {symbol}")
        print(f"{'='*60}")
        
        # 獲取當前價格
        print(f"Getting current price for {symbol}...")
        self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
        time.sleep(3)
        
        current_price = None
        if symbol in self.dwx._Market_Data_DB and self.dwx._Market_Data_DB[symbol]:
            data = self.dwx._Market_Data_DB[symbol]
            if data:
                latest_timestamp = list(data.keys())[-1]
                bid, ask = data[latest_timestamp]
                current_price = ask  # Use ask for buy
                print(f"Current price: Bid={bid:.2f}, Ask={ask:.2f}")
        
        if not current_price:
            print("[ERROR] Cannot get current price")
            self.test_results.append({'test': 'Open Position', 'status': 'FAILED'})
            return None
        
        # 發送買入指令
        print(f"\nSending BUY order...")
        print(f"  Symbol: {symbol}")
        print(f"  Volume: {volume}")
        print(f"  Price: {current_price:.2f}")
        
        # 計算停損停利
        sl = current_price - 100  # 100點停損
        tp = current_price + 200  # 200點停利
        
        order = {
            '_action': 'OPEN',
            '_type': 0,  # 0=BUY, 1=SELL
            '_symbol': symbol,
            '_price': 0,  # 0 for market order
            '_SL': sl,
            '_TP': tp,
            '_lots': volume,
            '_comment': 'Test Buy Order',
            '_magic': 123456
        }
        
        self.dwx._DWX_MTX_NEW_TRADE_(order)
        time.sleep(3)
        
        # 檢查開倉結果
        if hasattr(self.dwx, 'open_orders_DB') and self.dwx.open_orders_DB:
            print("[SUCCESS] Position opened successfully")
            for ticket, order_info in self.dwx.open_orders_DB.items():
                print(f"  Ticket: {ticket}")
                print(f"  Type: {'BUY' if order_info.get('type') == 0 else 'SELL'}")
                print(f"  Open Price: {order_info.get('open_price', 0):.2f}")
                
                self.test_results.append({
                    'test': 'Open Position',
                    'status': 'PASSED',
                    'ticket': ticket
                })
                return ticket
        else:
            print("[WARNING] Position may have opened but not confirmed")
            self.test_results.append({'test': 'Open Position', 'status': 'UNCERTAIN'})
            return None
    
    def test_query_positions(self):
        """測試3: 查詢持倉"""
        print(f"\n{'='*60}")
        print(f"TEST 3: Query Open Positions")
        print(f"{'='*60}")
        
        self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'open_orders_DB') and self.dwx.open_orders_DB:
            print(f"Found {len(self.dwx.open_orders_DB)} open positions:")
            
            for ticket, order_info in self.dwx.open_orders_DB.items():
                print(f"\nPosition {ticket}:")
                print(f"  Symbol: {order_info.get('symbol', 'N/A')}")
                print(f"  Type: {'BUY' if order_info.get('type') == 0 else 'SELL'}")
                print(f"  Volume: {order_info.get('lots', 0)}")
                print(f"  Open Price: {order_info.get('open_price', 0)}")
                print(f"  Current Price: {order_info.get('price_current', 0)}")
                print(f"  Profit: ${order_info.get('profit', 0):.2f}")
                print(f"  SL: {order_info.get('SL', 0)}")
                print(f"  TP: {order_info.get('TP', 0)}")
            
            self.test_results.append({
                'test': 'Query Positions',
                'status': 'PASSED',
                'count': len(self.dwx.open_orders_DB)
            })
            return True
        else:
            print("No open positions found")
            self.test_results.append({'test': 'Query Positions', 'status': 'PASSED'})
            return True
    
    def test_close_position(self, ticket=None):
        """測試4: 平倉測試"""
        print(f"\n{'='*60}")
        print(f"TEST 4: Close Position")
        print(f"{'='*60}")
        
        if not ticket:
            # 獲取第一個開倉
            self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            time.sleep(2)
            
            if hasattr(self.dwx, 'open_orders_DB') and self.dwx.open_orders_DB:
                ticket = list(self.dwx.open_orders_DB.keys())[0]
                print(f"Found position to close: {ticket}")
            else:
                print("No open positions to close")
                self.test_results.append({'test': 'Close Position', 'status': 'SKIPPED'})
                return False
        
        print(f"Closing position {ticket}...")
        self.dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(ticket)
        time.sleep(3)
        
        # 驗證平倉
        self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'open_orders_DB') and ticket not in self.dwx.open_orders_DB:
            print(f"[SUCCESS] Position {ticket} closed successfully")
            self.test_results.append({'test': 'Close Position', 'status': 'PASSED'})
            return True
        else:
            print(f"[ERROR] Failed to close position {ticket}")
            self.test_results.append({'test': 'Close Position', 'status': 'FAILED'})
            return False
    
    def test_modify_order(self, ticket=None):
        """測試5: 修改訂單（停損/停利）"""
        print(f"\n{'='*60}")
        print(f"TEST 5: Modify Order (SL/TP)")
        print(f"{'='*60}")
        
        if not ticket:
            # 獲取第一個開倉
            self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            time.sleep(2)
            
            if hasattr(self.dwx, 'open_orders_DB') and self.dwx.open_orders_DB:
                ticket = list(self.dwx.open_orders_DB.keys())[0]
                order_info = self.dwx.open_orders_DB[ticket]
                print(f"Found position to modify: {ticket}")
                
                # 計算新的停損停利
                current_price = order_info.get('price_current', 0)
                new_sl = current_price - 50
                new_tp = current_price + 100
                
                print(f"Modifying SL to {new_sl:.2f}, TP to {new_tp:.2f}")
                
                self.dwx._DWX_MTX_MODIFY_TRADE_BY_TICKET_(ticket, new_sl, new_tp)
                time.sleep(3)
                
                # 驗證修改
                self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
                time.sleep(2)
                
                if hasattr(self.dwx, 'open_orders_DB') and ticket in self.dwx.open_orders_DB:
                    updated_order = self.dwx.open_orders_DB[ticket]
                    if abs(updated_order.get('SL', 0) - new_sl) < 1:
                        print("[SUCCESS] Order modified successfully")
                        self.test_results.append({'test': 'Modify Order', 'status': 'PASSED'})
                        return True
                
                print("[ERROR] Failed to modify order")
                self.test_results.append({'test': 'Modify Order', 'status': 'FAILED'})
                return False
            else:
                print("No open positions to modify")
                self.test_results.append({'test': 'Modify Order', 'status': 'SKIPPED'})
                return False
    
    def run_all_tests(self):
        """執行所有測試"""
        print("\n" + "="*60)
        print(" MT4 Trading Function Test Suite ")
        print("="*60)
        print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(" WARNING: This will execute REAL trades on your account!")
        print("="*60)
        
        # 測試序列
        self.test_account_info()
        
        # 開倉測試 (使用最小手數)
        ticket = self.test_open_position('BTCUSD', 0.01)
        
        # 查詢持倉
        self.test_query_positions()
        
        # 修改訂單
        if ticket:
            self.test_modify_order(ticket)
        
        # 平倉測試
        if ticket:
            self.test_close_position(ticket)
        
        # 最終報告
        self.print_test_report()
    
    def print_test_report(self):
        """打印測試報告"""
        print("\n" + "="*60)
        print(" Test Report ")
        print("="*60)
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAILED')
        skipped = sum(1 for r in self.test_results if r['status'] == 'SKIPPED')
        
        for result in self.test_results:
            status_icon = '[PASS]' if result['status'] == 'PASSED' else '[FAIL]' if result['status'] == 'FAILED' else '[SKIP]'
            print(f"{status_icon} {result['test']}: {result['status']}")
        
        print(f"\nSummary:")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        print(f"  Total: {len(self.test_results)}")
        
        # 保存報告
        with open(f'trading_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        if failed == 0:
            print("\n[SUCCESS] All tests passed! MT4 trading functions working correctly.")
        else:
            print(f"\n[WARNING] {failed} tests failed. Please check the errors above.")
    
    def disconnect(self):
        """斷開連接"""
        if self.dwx:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print("Disconnected")

def main():
    """主函數"""
    tester = MT4TradingTester()
    
    try:
        if tester.connect():
            # 詢問用戶確認
            print("\n" + "!"*60)
            print(" WARNING: This test will execute REAL trades!")
            print(" Make sure you are using a DEMO account!")
            print("!"*60)
            
            response = input("\nType 'YES' to continue, or anything else to abort: ")
            
            if response.upper() == 'YES':
                tester.run_all_tests()
            else:
                print("Test aborted by user")
                # 只測試不會交易的功能
                tester.test_account_info()
                tester.test_query_positions()
                tester.print_test_report()
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        tester.disconnect()

if __name__ == "__main__":
    main()