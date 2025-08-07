#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick BTC Symbol Test"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
import time
from datetime import datetime

print("\n" + "="*50)
print(" Quick BTC/Crypto Symbol Test ")
print("="*50)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)

# Connect
print("\nConnecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='BTCTest',
    _verbose=False,
    _poll_timeout=1000
)

time.sleep(2)

# Get account info
print("Getting account info...")
dwx._DWX_MTX_GET_ACCOUNT_INFO_()
time.sleep(2)

if hasattr(dwx, 'account_info_DB') and dwx.account_info_DB:
    for acc_num, info in dwx.account_info_DB.items():
        if info:
            print(f"Account: {acc_num}, Balance: ${info[0].get('account_balance', 0):.2f}")

# Test common crypto symbols
print("\nTesting crypto symbols:")
test_symbols = ['BTCUSD', 'Bitcoin', 'ETHUSD', 'Ethereum']

found_symbols = []
for symbol in test_symbols:
    print(f"\nTrying {symbol}...")
    dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
    time.sleep(2)
    
    # Check if we got data
    if dwx._Market_Data_DB:
        for sym, data in dwx._Market_Data_DB.items():
            if data:
                latest = list(data.keys())[-1]
                bid, ask = data[latest]
                print(f"  SUCCESS: {sym} = {bid:.2f} / {ask:.2f}")
                found_symbols.append(sym)

if not found_symbols:
    print("\nNo crypto symbols found. Testing forex...")
    dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('EURUSD')
    time.sleep(2)
    
    if dwx._Market_Data_DB:
        for sym, data in dwx._Market_Data_DB.items():
            if data:
                latest = list(data.keys())[-1]
                bid, ask = data[latest]
                print(f"  {sym} = {bid:.5f} / {ask:.5f}")

print("\n" + "="*50)
print(" Summary ")
print("="*50)
if found_symbols:
    print(f"Found crypto: {', '.join(found_symbols)}")
else:
    print("No crypto symbols available")
    print("Your broker may not offer crypto trading")
    print("or symbols may have different names")

# Cleanup
print("\nClosing connection...")
dwx._DWX_ZMQ_SHUTDOWN_()