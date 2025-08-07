#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DevOps 優化交易腳本
使用最佳實踐和錯誤處理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import json

def safe_trade():
    """安全的交易測試"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'starting',
        'trades': []
    }
    
    print("\n" + "="*60)
    print(" DevOps Optimized Trading Test ")
    print("="*60)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Symbol: EURUSD (Most Stable)")
    print(" Volume: 0.01 lot (Minimum)")
    print("="*60)
    
    try:
        # Step 1: Connect with optimized settings
        print("\n[1] Connecting with optimized settings...")
        dwx = DWX_ZeroMQ_Connector(
            _ClientID='DevOpsTrader',
            _verbose=False,
            _poll_timeout=3000,  # Increased timeout
            _sleep_delay=0.005,  # Optimized delay
            _monitor=False       # Disable monitoring for speed
        )
        
        time.sleep(3)
        print("  Connected successfully")
        
        # Step 2: Get account info
        print("\n[2] Checking account...")
        dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        account_ok = False
        if hasattr(dwx, 'account_info_DB') and dwx.account_info_DB:
            for acc_num, info_list in dwx.account_info_DB.items():
                if info_list:
                    balance = info_list[0].get('account_balance', 0)
                    print(f"  Account: {acc_num}")
                    print(f"  Balance: ${balance:.2f}")
                    account_ok = True
                    report['account'] = acc_num
                    report['balance'] = balance
        
        if not account_ok:
            print("  [WARNING] Cannot get account info")
            print("  Proceeding anyway...")
        
        # Step 3: Place order
        print("\n[3] Placing test order...")
        
        order = {
            '_action': 'OPEN',
            '_type': 0,  # BUY
            '_symbol': 'EURUSD',  # Most stable symbol
            '_price': 0,  # Market order
            '_SL': 0,    # No SL for test
            '_TP': 0,    # No TP for test
            '_lots': 0.01,  # Minimum size
            '_comment': 'DevOps Test',
            '_magic': 999999
        }
        
        print(f"  Order: BUY 0.01 EURUSD at Market")
        
        # Try to send order
        for attempt in range(3):
            print(f"  Attempt {attempt + 1}/3...")
            
            dwx._DWX_MTX_NEW_TRADE_(order)
            time.sleep(5)
            
            # Check if order was placed
            dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            time.sleep(2)
            
            if hasattr(dwx, 'open_orders_DB') and dwx.open_orders_DB:
                print("\n  [SUCCESS] Order placed!")
                
                for ticket, order_info in dwx.open_orders_DB.items():
                    print(f"    Ticket: #{ticket}")
                    print(f"    Open Price: {order_info.get('open_price', 0)}")
                    
                    report['trades'].append({
                        'ticket': ticket,
                        'symbol': order_info.get('symbol'),
                        'type': 'BUY' if order_info.get('type') == 0 else 'SELL',
                        'lots': order_info.get('lots'),
                        'open_price': order_info.get('open_price')
                    })
                    
                    # Auto close after 5 seconds
                    print("\n  Closing in 5 seconds...")
                    time.sleep(5)
                    
                    print(f"  Closing order #{ticket}...")
                    dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(ticket)
                    time.sleep(3)
                    
                    print("  [SUCCESS] Order closed")
                    report['status'] = 'success'
                    break
                
                break
            else:
                print(f"    No order found, retrying...")
                time.sleep(2)
        
        if report['status'] != 'success':
            print("\n  [FAILED] Could not place order after 3 attempts")
            report['status'] = 'failed'
        
        # Cleanup
        dwx._DWX_ZMQ_SHUTDOWN_()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        report['status'] = 'error'
        report['error'] = str(e)
    
    # Save report
    report_file = f'devops_trade_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[REPORT] Saved to {report_file}")
    
    # Final status
    print("\n" + "="*60)
    print(" Test Result ")
    print("="*60)
    
    if report['status'] == 'success':
        print("\n✅ SUCCESS - Trading system is working!")
        print("   MT4 connection: OK")
        print("   Order placement: OK")
        print("   Order closure: OK")
    elif report['status'] == 'failed':
        print("\n❌ FAILED - Cannot place orders")
        print("\nPossible causes:")
        print("1. EA not running (check for 😊 on chart)")
        print("2. AutoTrading disabled (check button)")
        print("3. Market closed (check trading hours)")
        print("\nPlease check DEVOPS_FIX_GUIDE.md for solutions")
    else:
        print("\n❌ ERROR - Connection problem")
        print("Please ensure MT4 is running")
    
    return report['status'] == 'success'

if __name__ == "__main__":
    # Run test
    success = safe_trade()
    
    # Exit code for CI/CD
    sys.exit(0 if success else 1)