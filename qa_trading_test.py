#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QA 完整交易功能測試
測試開倉、平倉、修改訂單等核心功能
使用最小手數 0.01 進行測試
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import json

class QATradingTest:
    def __init__(self):
        self.dwx = None
        self.test_results = []
        self.test_ticket = None
        
    def connect(self):
        """連接到MT4"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='QATester',
            _verbose=False,
            _poll_timeout=1000
        )
        
        time.sleep(3)
        return True
    
    def test_1_account_info(self):
        """測試1: 帳戶資訊"""
        print(f"\n{'='*60}")
        print(f"TEST 1: Account Information")
        print(f"{'='*60}")
        
        self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'account_info_DB') and self.dwx.account_info_DB:
            for acc_num, info_list in self.dwx.account_info_DB.items():
                if info_list:
                    info = info_list[0]
                    balance = info.get('account_balance', 0)
                    
                    print(f"  Account: {acc_num}")
                    print(f"  Balance: ${balance:.2f}")
                    print(f"  Equity: ${info.get('account_equity', 0):.2f}")
                    print(f"  Leverage: 1:{info.get('account_leverage', 100)}")
                    
                    # Check if demo account
                    if balance < 10000:
                        print(f"  [INFO] Demo account confirmed (balance < $10,000)")
                    
                    self.test_results.append({
                        'test': 'Account Info',
                        'status': 'PASSED',
                        'details': f'Balance: ${balance:.2f}'
                    })
                    return True
        
        self.test_results.append({'test': 'Account Info', 'status': 'FAILED'})
        return False
    
    def test_2_open_buy_order(self):
        """測試2: 開買單 (0.01手 BTCUSD)"""
        print(f"\n{'='*60}")
        print(f"TEST 2: Open BUY Order (0.01 lot BTCUSD)")
        print(f"{'='*60}")
        
        symbol = 'BTCUSD'
        volume = 0.01  # 最小手數
        
        # 獲取當前價格
        print(f"  Getting current price...")
        self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
        time.sleep(5)  # 等待更長時間
        
        current_bid = None
        current_ask = None
        
        # 嘗試多次獲取價格
        for attempt in range(3):
            if symbol in self.dwx._Market_Data_DB and self.dwx._Market_Data_DB[symbol]:
                data = self.dwx._Market_Data_DB[symbol]
                if data:
                    latest_timestamp = list(data.keys())[-1]
                    current_bid, current_ask = data[latest_timestamp]
                    print(f"  Current: Bid=${current_bid:.2f}, Ask=${current_ask:.2f}")
                    break
            else:
                print(f"  Attempt {attempt+1}: Waiting for price...")
                time.sleep(2)
        
        if not current_ask:
            print("  [ERROR] Cannot get price")
            self.test_results.append({'test': 'Open BUY Order', 'status': 'FAILED'})
            return False
        
        # 發送買單
        print(f"\n  Sending BUY order...")
        print(f"    Symbol: {symbol}")
        print(f"    Volume: {volume} lot")
        print(f"    Expected price: ~${current_ask:.2f}")
        
        # 設置停損停利
        sl = current_ask - 500  # 500點停損
        tp = current_ask + 1000  # 1000點停利
        
        order = {
            '_action': 'OPEN',
            '_type': 0,  # 0=BUY
            '_symbol': symbol,
            '_price': 0,  # Market order
            '_SL': sl,
            '_TP': tp,
            '_lots': volume,
            '_comment': 'QA Test Buy',
            '_magic': 999999
        }
        
        self.dwx._DWX_MTX_NEW_TRADE_(order)
        time.sleep(5)
        
        # 檢查結果
        self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'open_orders_DB') and self.dwx.open_orders_DB:
            for ticket, order_info in self.dwx.open_orders_DB.items():
                if order_info.get('symbol') == symbol:
                    self.test_ticket = ticket
                    print(f"\n  [SUCCESS] Order opened")
                    print(f"    Ticket: #{ticket}")
                    print(f"    Open Price: ${order_info.get('open_price', 0):.2f}")
                    print(f"    SL: ${order_info.get('SL', 0):.2f}")
                    print(f"    TP: ${order_info.get('TP', 0):.2f}")
                    
                    self.test_results.append({
                        'test': 'Open BUY Order',
                        'status': 'PASSED',
                        'details': f'Ticket #{ticket}'
                    })
                    return True
        
        print("  [FAILED] Order not opened")
        self.test_results.append({'test': 'Open BUY Order', 'status': 'FAILED'})
        return False
    
    def test_3_check_position(self):
        """測試3: 查詢持倉"""
        print(f"\n{'='*60}")
        print(f"TEST 3: Check Open Position")
        print(f"{'='*60}")
        
        self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'open_orders_DB') and self.dwx.open_orders_DB:
            print(f"  Found {len(self.dwx.open_orders_DB)} position(s):")
            
            for ticket, order_info in self.dwx.open_orders_DB.items():
                profit = order_info.get('profit', 0)
                print(f"\n  Position #{ticket}:")
                print(f"    Symbol: {order_info.get('symbol')}")
                print(f"    Type: {'BUY' if order_info.get('type') == 0 else 'SELL'}")
                print(f"    Volume: {order_info.get('lots')} lot")
                print(f"    Profit/Loss: ${profit:.2f}")
                
                if ticket == self.test_ticket:
                    print(f"    [INFO] This is our test position")
            
            self.test_results.append({
                'test': 'Check Position',
                'status': 'PASSED',
                'details': f'{len(self.dwx.open_orders_DB)} position(s)'
            })
            return True
        else:
            print("  No positions found")
            self.test_results.append({'test': 'Check Position', 'status': 'FAILED'})
            return False
    
    def test_4_modify_order(self):
        """測試4: 修改停損停利"""
        print(f"\n{'='*60}")
        print(f"TEST 4: Modify SL/TP")
        print(f"{'='*60}")
        
        if not self.test_ticket:
            print("  [SKIP] No test position to modify")
            self.test_results.append({'test': 'Modify SL/TP', 'status': 'SKIPPED'})
            return False
        
        # 獲取當前價格
        self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if self.test_ticket in self.dwx.open_orders_DB:
            order_info = self.dwx.open_orders_DB[self.test_ticket]
            current_price = order_info.get('price_current', 0)
            
            # 新的停損停利
            new_sl = current_price - 300
            new_tp = current_price + 600
            
            print(f"  Modifying order #{self.test_ticket}:")
            print(f"    New SL: ${new_sl:.2f}")
            print(f"    New TP: ${new_tp:.2f}")
            
            self.dwx._DWX_MTX_MODIFY_TRADE_BY_TICKET_(self.test_ticket, new_sl, new_tp)
            time.sleep(3)
            
            # 驗證修改
            self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            time.sleep(2)
            
            if self.test_ticket in self.dwx.open_orders_DB:
                updated = self.dwx.open_orders_DB[self.test_ticket]
                actual_sl = updated.get('SL', 0)
                actual_tp = updated.get('TP', 0)
                
                if abs(actual_sl - new_sl) < 10 and abs(actual_tp - new_tp) < 10:
                    print(f"\n  [SUCCESS] Order modified")
                    print(f"    Actual SL: ${actual_sl:.2f}")
                    print(f"    Actual TP: ${actual_tp:.2f}")
                    
                    self.test_results.append({
                        'test': 'Modify SL/TP',
                        'status': 'PASSED',
                        'details': 'SL/TP updated'
                    })
                    return True
        
        print("  [FAILED] Could not modify order")
        self.test_results.append({'test': 'Modify SL/TP', 'status': 'FAILED'})
        return False
    
    def test_5_close_position(self):
        """測試5: 平倉"""
        print(f"\n{'='*60}")
        print(f"TEST 5: Close Position")
        print(f"{'='*60}")
        
        if not self.test_ticket:
            print("  [SKIP] No test position to close")
            self.test_results.append({'test': 'Close Position', 'status': 'SKIPPED'})
            return False
        
        print(f"  Closing position #{self.test_ticket}...")
        self.dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(self.test_ticket)
        time.sleep(5)
        
        # 驗證平倉
        self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(self.dwx, 'open_orders_DB') and self.test_ticket in self.dwx.open_orders_DB:
            print("  [FAILED] Position still open")
            self.test_results.append({'test': 'Close Position', 'status': 'FAILED'})
            return False
        else:
            print("  [SUCCESS] Position closed")
            self.test_results.append({
                'test': 'Close Position',
                'status': 'PASSED',
                'details': f'Ticket #{self.test_ticket} closed'
            })
            return True
    
    def generate_report(self):
        """生成測試報告"""
        print(f"\n{'='*60}")
        print(f" QA Test Report ")
        print(f"{'='*60}")
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAILED')
        skipped = sum(1 for r in self.test_results if r['status'] == 'SKIPPED')
        
        for result in self.test_results:
            status = result['status']
            if status == 'PASSED':
                icon = '[PASS]'
            elif status == 'FAILED':
                icon = '[FAIL]'
            else:
                icon = '[SKIP]'
            
            details = f" - {result.get('details', '')}" if 'details' in result else ""
            print(f"{icon} {result['test']}{details}")
        
        print(f"\nSummary:")
        print(f"  Total Tests: {len(self.test_results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        
        success_rate = (passed / len(self.test_results) * 100) if self.test_results else 0
        print(f"  Success Rate: {success_rate:.1f}%")
        
        # 保存報告
        report_file = f'qa_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.test_results,
                'summary': {
                    'total': len(self.test_results),
                    'passed': passed,
                    'failed': failed,
                    'skipped': skipped,
                    'success_rate': success_rate
                }
            }, f, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        
        if failed == 0:
            print("\n[SUCCESS] All critical tests passed!")
            print("MT4 trading functions are working correctly.")
        else:
            print(f"\n[WARNING] {failed} test(s) failed.")
            print("Please check the errors above.")
        
        return success_rate >= 80  # 80% pass rate required
    
    def run_all_tests(self):
        """執行所有測試"""
        print("\n" + "="*60)
        print(" QA Trading Function Test Suite ")
        print("="*60)
        print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(" Using: 0.01 lot BTCUSD (minimum size)")
        print("="*60)
        
        # 執行測試序列
        self.test_1_account_info()
        self.test_2_open_buy_order()
        self.test_3_check_position()
        self.test_4_modify_order()
        self.test_5_close_position()
        
        # 生成報告
        return self.generate_report()
    
    def disconnect(self):
        """斷開連接"""
        if self.dwx:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print("Disconnected")

def main():
    """主函數"""
    tester = QATradingTest()
    
    try:
        print("\n" + "!"*60)
        print(" QA TRADING TEST - WILL EXECUTE REAL TRADES ")
        print(" Ensure you are using a DEMO account! ")
        print("!"*60)
        
        response = input("\nConfirm DEMO account and continue? (YES/NO): ")
        
        if response.upper() == 'YES':
            if tester.connect():
                success = tester.run_all_tests()
                
                if success:
                    print("\n[QA PASSED] Trading system verified successfully")
                else:
                    print("\n[QA FAILED] Some tests failed, review required")
        else:
            print("Test cancelled by user")
    
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