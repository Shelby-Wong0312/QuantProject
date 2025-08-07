#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檢查並清理現有訂單
QA 任務：確保測試環境乾淨
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time

print("\n" + "="*60)
print(" MT4 Order Check and Clean ")
print("="*60)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Connect
print("\nConnecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='OrderChecker',
    _verbose=False,
    _poll_timeout=1000
)

time.sleep(2)

# Get all open trades
print("\nChecking for open orders...")
dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
time.sleep(2)

if hasattr(dwx, 'open_orders_DB') and dwx.open_orders_DB:
    print(f"\nFound {len(dwx.open_orders_DB)} open orders:")
    
    for ticket, order_info in dwx.open_orders_DB.items():
        print(f"\n  Order #{ticket}:")
        print(f"    Symbol: {order_info.get('symbol', 'N/A')}")
        order_type = order_info.get('type', -1)
        if order_type == 0:
            type_str = "BUY"
        elif order_type == 1:
            type_str = "SELL"
        elif order_type == 2:
            type_str = "BUY LIMIT"
        elif order_type == 3:
            type_str = "SELL LIMIT"
        elif order_type == 4:
            type_str = "BUY STOP"
        elif order_type == 5:
            type_str = "SELL STOP"
        else:
            type_str = f"TYPE {order_type}"
        
        print(f"    Type: {type_str}")
        print(f"    Volume: {order_info.get('lots', 0)}")
        print(f"    Open Price: {order_info.get('open_price', 0)}")
        print(f"    Open Time: {order_info.get('open_time', 'N/A')}")
        print(f"    Comment: {order_info.get('comment', '')}")
    
    # Ask to clean
    response = input("\nDelete all pending orders? (YES/NO): ")
    
    if response.upper() == 'YES':
        for ticket in dwx.open_orders_DB.keys():
            print(f"  Deleting order #{ticket}...")
            dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(ticket)
            time.sleep(1)
        
        # Verify
        time.sleep(2)
        dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        if hasattr(dwx, 'open_orders_DB') and dwx.open_orders_DB:
            print(f"\n[WARNING] Still have {len(dwx.open_orders_DB)} orders")
        else:
            print("\n[SUCCESS] All orders deleted")
    else:
        print("\nKeeping existing orders")
else:
    print("\n[SUCCESS] No open orders found - environment is clean")

# Disconnect
print("\nDisconnecting...")
dwx._DWX_ZMQ_SHUTDOWN_()
print("Done!")