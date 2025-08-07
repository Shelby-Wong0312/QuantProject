#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QA 簡化交易測試 - 不依賴價格數據
直接使用市價單測試
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import json

print("\n" + "="*60)
print(" QA Simple Trade Test ")
print("="*60)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Connect
print("\n[STEP 1] Connecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='QASimple',
    _verbose=True,  # 開啟詳細輸出
    _poll_timeout=1000
)

time.sleep(3)

# Test 1: Account Info
print("\n[TEST 1] Account Info")
print("-"*40)
dwx._DWX_MTX_GET_ACCOUNT_INFO_()
time.sleep(2)

if hasattr(dwx, 'account_info_DB') and dwx.account_info_DB:
    for acc_num, info_list in dwx.account_info_DB.items():
        if info_list:
            info = info_list[0]
            print(f"Account: {acc_num}")
            print(f"Balance: ${info.get('account_balance', 0):.2f}")
            print(f"[PASS] Account info retrieved")
else:
    print("[FAIL] Cannot get account info")

# Test 2: Simple Market Buy Order
print("\n[TEST 2] Market Buy Order")
print("-"*40)
print("Sending market BUY order for 0.01 lot BTCUSD...")

order = {
    '_action': 'OPEN',
    '_type': 0,  # 0=BUY
    '_symbol': 'BTCUSD',
    '_price': 0,  # 0 = market order
    '_SL': 0,     # No SL initially
    '_TP': 0,     # No TP initially  
    '_lots': 0.01,
    '_comment': 'QA Simple Test',
    '_magic': 123123
}

print(f"Order details: {order}")
dwx._DWX_MTX_NEW_TRADE_(order)
time.sleep(5)

# Check if order was opened
print("\n[TEST 3] Check Open Trades")
print("-"*40)
dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
time.sleep(2)

test_ticket = None
if hasattr(dwx, 'open_orders_DB') and dwx.open_orders_DB:
    print(f"Found {len(dwx.open_orders_DB)} open trade(s):")
    
    for ticket, order_info in dwx.open_orders_DB.items():
        print(f"\nTicket: #{ticket}")
        print(f"  Symbol: {order_info.get('symbol')}")
        print(f"  Type: {'BUY' if order_info.get('type') == 0 else 'SELL'}")
        print(f"  Volume: {order_info.get('lots')} lot")
        print(f"  Open Price: ${order_info.get('open_price', 0):.2f}")
        print(f"  Comment: {order_info.get('comment')}")
        
        if order_info.get('comment') == 'QA Simple Test':
            test_ticket = ticket
            print(f"[PASS] Our test order found!")
else:
    print("[WARNING] No open trades found")

# Test 4: Close Trade
if test_ticket:
    print(f"\n[TEST 4] Close Trade #{test_ticket}")
    print("-"*40)
    
    response = input("Close the test position? (YES/NO): ")
    
    if response.upper() == 'YES':
        print(f"Closing position #{test_ticket}...")
        dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(test_ticket)
        time.sleep(5)
        
        # Verify closure
        dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(dwx, 'open_orders_DB') and test_ticket not in dwx.open_orders_DB:
            print("[PASS] Position closed successfully")
        else:
            print("[WARNING] Position may still be open")
    else:
        print("Keeping position open (remember to close manually!)")
else:
    print("\n[SKIP] No test position to close")

# Final Report
print("\n" + "="*60)
print(" Test Summary ")
print("="*60)

test_results = {
    'timestamp': datetime.now().isoformat(),
    'account_test': 'PASSED' if hasattr(dwx, 'account_info_DB') else 'FAILED',
    'order_opened': 'PASSED' if test_ticket else 'FAILED',
    'test_ticket': test_ticket
}

print(f"Account Test: {test_results['account_test']}")
print(f"Order Test: {test_results['order_opened']}")
if test_ticket:
    print(f"Test Ticket: #{test_ticket}")

# Save results
with open(f'qa_simple_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(test_results, f, indent=2)

# Disconnect
print("\nDisconnecting...")
dwx._DWX_ZMQ_SHUTDOWN_()
print("Done!")