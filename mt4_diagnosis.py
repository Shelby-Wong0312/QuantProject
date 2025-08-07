#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4 交易功能診斷工具
DevOps 任務：診斷並修復交易執行問題
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import zmq

class MT4Diagnostics:
    def __init__(self):
        self.issues = []
        self.recommendations = []
        
    def test_1_zmq_connection(self):
        """測試 ZeroMQ 連接"""
        print("\n[TEST 1] ZeroMQ Connection Test")
        print("-"*50)
        
        try:
            context = zmq.Context()
            
            # Test PUSH socket
            push = context.socket(zmq.PUSH)
            push.connect("tcp://localhost:32768")
            print("  PUSH socket (32768): OK")
            
            # Test PULL socket  
            pull = context.socket(zmq.PULL)
            pull.connect("tcp://localhost:32769")
            pull.setsockopt(zmq.RCVTIMEO, 1000)
            print("  PULL socket (32769): OK")
            
            # Test SUB socket
            sub = context.socket(zmq.SUB)
            sub.connect("tcp://localhost:32770")
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            print("  SUB socket (32770): OK")
            
            # Send heartbeat
            push.send_string("HEARTBEAT;")
            try:
                response = pull.recv_string()
                print(f"  Heartbeat response: {response[:50]}")
                print("\n[PASS] ZeroMQ connection working")
            except zmq.Again:
                print("\n[FAIL] No heartbeat response - EA may not be running")
                self.issues.append("EA not responding to heartbeat")
                self.recommendations.append("Check if DWX EA is running in MT4")
            
            # Cleanup
            push.close()
            pull.close()
            sub.close()
            context.term()
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] ZeroMQ connection failed: {e}")
            self.issues.append(f"ZeroMQ error: {e}")
            return False
    
    def test_2_dwx_connector(self):
        """測試 DWX 連接器"""
        print("\n[TEST 2] DWX Connector Test")
        print("-"*50)
        
        try:
            dwx = DWX_ZeroMQ_Connector(
                _ClientID='Diagnostics',
                _verbose=True,
                _poll_timeout=1000
            )
            
            time.sleep(3)
            
            # Test account info
            dwx._DWX_MTX_GET_ACCOUNT_INFO_()
            time.sleep(2)
            
            if hasattr(dwx, 'account_info_DB') and dwx.account_info_DB:
                print("\n[PASS] DWX connector can retrieve account info")
            else:
                print("\n[WARNING] Cannot get account info")
                self.issues.append("Account info retrieval failed")
            
            dwx._DWX_ZMQ_SHUTDOWN_()
            return True
            
        except Exception as e:
            print(f"\n[ERROR] DWX connector failed: {e}")
            self.issues.append(f"DWX connector error: {e}")
            return False
    
    def test_3_symbol_availability(self):
        """測試交易符號可用性"""
        print("\n[TEST 3] Symbol Trading Availability")
        print("-"*50)
        
        try:
            dwx = DWX_ZeroMQ_Connector(
                _ClientID='SymbolTest',
                _verbose=False,
                _poll_timeout=1000
            )
            
            time.sleep(3)
            
            # Test different symbols
            test_symbols = ['EURUSD', 'GOLD', 'BTCUSD', 'XRPUSD']
            tradeable = []
            
            for symbol in test_symbols:
                print(f"\n  Testing {symbol}...")
                
                # Try to get market info
                dwx._DWX_MTX_SEND_MARKETDATA_REQUEST_(symbol)
                time.sleep(2)
                
                # Check if we can subscribe
                dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
                time.sleep(3)
                
                if symbol in dwx._Market_Data_DB and dwx._Market_Data_DB[symbol]:
                    print(f"    [OK] {symbol} is available for trading")
                    tradeable.append(symbol)
                else:
                    print(f"    [NO] {symbol} not available or no data")
                
                dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
            
            if not tradeable:
                self.issues.append("No symbols available for trading")
                self.recommendations.append("Check Market Watch in MT4")
            else:
                print(f"\n[INFO] Tradeable symbols: {', '.join(tradeable)}")
                if 'BTCUSD' not in tradeable:
                    self.recommendations.append("Use EURUSD or GOLD for testing instead of BTCUSD")
            
            dwx._DWX_ZMQ_SHUTDOWN_()
            return len(tradeable) > 0
            
        except Exception as e:
            print(f"\n[ERROR] Symbol test failed: {e}")
            return False
    
    def test_4_simple_order_test(self):
        """測試簡單下單（使用 EURUSD）"""
        print("\n[TEST 4] Simple Order Test (EURUSD)")
        print("-"*50)
        
        try:
            dwx = DWX_ZeroMQ_Connector(
                _ClientID='OrderTest',
                _verbose=True,
                _poll_timeout=3000  # Longer timeout
            )
            
            time.sleep(3)
            
            # Use EURUSD which is more likely to work
            symbol = 'EURUSD'
            print(f"\n  Testing order on {symbol}...")
            
            order = {
                '_action': 'OPEN',
                '_type': 0,  # BUY
                '_symbol': symbol,
                '_price': 0,  # Market order
                '_SL': 0,
                '_TP': 0,
                '_lots': 0.01,
                '_comment': 'Diagnostic Test',
                '_magic': 777777
            }
            
            print(f"  Sending order: {order}")
            dwx._DWX_MTX_NEW_TRADE_(order)
            time.sleep(5)
            
            # Check if order was placed
            dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            time.sleep(2)
            
            if hasattr(dwx, 'open_orders_DB') and dwx.open_orders_DB:
                print("\n[SUCCESS] Order placed successfully!")
                
                # Close the test order
                for ticket in dwx.open_orders_DB.keys():
                    print(f"  Closing test order #{ticket}...")
                    dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(ticket)
                    time.sleep(2)
            else:
                print("\n[FAIL] Order was not placed")
                self.issues.append("Cannot place orders")
                self.recommendations.append("Check EA settings: Allow Live Trading must be ON")
                self.recommendations.append("Check Tools > Options > Expert Advisors > Allow automated trading")
            
            dwx._DWX_ZMQ_SHUTDOWN_()
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Order test failed: {e}")
            self.issues.append(f"Order test error: {e}")
            return False
    
    def generate_report(self):
        """生成診斷報告"""
        print("\n" + "="*60)
        print(" DIAGNOSTIC REPORT ")
        print("="*60)
        
        if self.issues:
            print("\n[ISSUES FOUND]")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n[NO ISSUES] All tests passed")
        
        if self.recommendations:
            print("\n[RECOMMENDATIONS]")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n[MT4 CHECKLIST]")
        print("Please verify in MT4:")
        print("  1. EA is attached to chart (smiley face icon)")
        print("  2. AutoTrading button is ON (green)")
        print("  3. Tools > Options > Expert Advisors:")
        print("     - [x] Allow automated trading")
        print("     - [x] Allow DLL imports")
        print("  4. EA Properties (F7 on chart):")
        print("     - [x] Allow live trading")
        print("     - [x] Allow DLL imports")
        print("  5. Market Watch contains desired symbols")
        print("  6. Account has sufficient margin")
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'issues': self.issues,
            'recommendations': self.recommendations
        }
        
        import json
        with open(f'mt4_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: mt4_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return len(self.issues) == 0

def main():
    print("\n" + "="*60)
    print(" MT4 Trading System Diagnostics ")
    print("="*60)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Purpose: Diagnose and fix trading execution issues")
    print("="*60)
    
    diag = MT4Diagnostics()
    
    # Run all tests
    diag.test_1_zmq_connection()
    diag.test_2_dwx_connector()
    diag.test_3_symbol_availability()
    
    # Only test orders if other tests pass
    if len(diag.issues) == 0:
        response = input("\nTest order placement? (YES/NO): ")
        if response.upper() == 'YES':
            diag.test_4_simple_order_test()
    
    # Generate report
    success = diag.generate_report()
    
    if success:
        print("\n[DIAGNOSIS COMPLETE] System is working correctly")
    else:
        print("\n[DIAGNOSIS COMPLETE] Issues found - see recommendations above")
    
    print("\n" + "="*60)
    print(" Next Steps ")
    print("="*60)
    print("1. Fix any MT4 configuration issues listed above")
    print("2. Restart MT4 and reattach EA if needed")
    print("3. Run 'python qa_trading_test.py' to verify fixes")

if __name__ == "__main__":
    main()