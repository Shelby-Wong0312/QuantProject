#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct test of DWX connection - minimal code
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
import time

print("\n" + "="*60)
print(" Direct DWX Connection Test ")
print("="*60)

# Initialize connector
print("\nInitializing DWX Connector...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='TestClient',
    _host='localhost',
    _protocol='tcp',
    _PUSH_PORT=32768,
    _PULL_PORT=32769,
    _SUB_PORT=32770,
    _verbose=True,
    _poll_timeout=1000,
    _sleep_delay=0.001
)

print("\n[OK] Connector created")
time.sleep(2)

# Test 1: Send HEARTBEAT
print("\n1. Sending HEARTBEAT...")
dwx._DWX_ZMQ_HEARTBEAT_()
time.sleep(2)

# Test 2: Get Account Info
print("\n2. Requesting Account Info...")
dwx._DWX_MTX_GET_ACCOUNT_INFO_()
time.sleep(3)

# Check if we got any response
if hasattr(dwx, 'account_info_DB') and dwx.account_info_DB:
    print(f"[SUCCESS] Account info received: {dwx.account_info_DB}")
else:
    print("[NO DATA] No account info received")

# Test 3: Subscribe to EURUSD
print("\n3. Subscribing to EURUSD...")
dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('EURUSD')
time.sleep(2)

# Test 4: Check for market data
print("\n4. Waiting for market data (5 seconds)...")
for i in range(5):
    time.sleep(1)
    if dwx._Market_Data_DB:
        print(f"[SUCCESS] Market data received: {list(dwx._Market_Data_DB.keys())}")
        for symbol, data in dwx._Market_Data_DB.items():
            latest_timestamp = list(data.keys())[-1] if data else None
            if latest_timestamp:
                print(f"  {symbol}: {data[latest_timestamp]}")
        break
    else:
        print(f"  {i+1}/5 - No data yet...")

# Test 5: Get open trades
print("\n5. Getting open trades...")
dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
time.sleep(2)

# Final check
print("\n" + "="*60)
print(" Test Results ")
print("="*60)

if dwx._Market_Data_DB:
    print("[SUCCESS] Connection is working - received market data")
elif hasattr(dwx, 'account_info_DB') and dwx.account_info_DB:
    print("[PARTIAL] Connection established but no market data")
else:
    print("[FAILED] No response from MT4")
    print("\nPlease verify in MT4:")
    print("1. Expert tab shows 'DWX Server' messages")
    print("2. No error messages in Journal tab")
    print("3. AutoTrading is enabled (green button)")

print("\nClosing connection...")
dwx._DWX_ZMQ_SHUTDOWN_()