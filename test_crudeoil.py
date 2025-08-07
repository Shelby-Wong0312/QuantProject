#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test CRUDEOIL Symbol
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time

print("\n" + "="*60)
print(" Testing CRUDEOIL Symbol ")
print("="*60)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Connect
print("\nConnecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='CRUDEOILTester',
    _verbose=False,
    _poll_timeout=1000
)

time.sleep(2)

# Test CRUDEOIL
print("\nTesting CRUDEOIL...")
dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('CRUDEOIL')

print("Waiting for data (5 seconds)...")
time.sleep(5)

# Check for data
if 'CRUDEOIL' in dwx._Market_Data_DB and dwx._Market_Data_DB['CRUDEOIL']:
    data = dwx._Market_Data_DB['CRUDEOIL']
    if data:
        print("\n[SUCCESS] CRUDEOIL is available!")
        
        # Get latest price
        latest_timestamp = list(data.keys())[-1]
        bid, ask = data[latest_timestamp]
        
        print(f"  Bid: ${bid:.2f}")
        print(f"  Ask: ${ask:.2f}")
        print(f"  Spread: ${ask-bid:.2f}")
        print(f"  Ticks received: {len(data)}")
        
        # Collect more data
        print("\nCollecting 10 seconds of CRUDEOIL data...")
        time.sleep(10)
        
        if len(dwx._Market_Data_DB['CRUDEOIL']) > len(data):
            print(f"  Total ticks: {len(dwx._Market_Data_DB['CRUDEOIL'])}")
            
            # Get price range
            all_prices = []
            for timestamp, (b, a) in dwx._Market_Data_DB['CRUDEOIL'].items():
                all_prices.append(b)
            
            print(f"  Price range: ${min(all_prices):.2f} - ${max(all_prices):.2f}")
        
        print("\n[CONFIRMED] WTI Crude Oil is available as: CRUDEOIL")
    else:
        print("\n[ERROR] CRUDEOIL subscribed but no data received")
else:
    print("\n[NOT FOUND] CRUDEOIL is not available")
    print("Please verify the symbol name with your broker")

# Also test BTCUSD for comparison
print("\n" + "-"*60)
print("Testing BTCUSD for comparison...")
dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('BTCUSD')
time.sleep(3)

if 'BTCUSD' in dwx._Market_Data_DB and dwx._Market_Data_DB['BTCUSD']:
    data = dwx._Market_Data_DB['BTCUSD']
    if data:
        latest_timestamp = list(data.keys())[-1]
        bid, ask = data[latest_timestamp]
        print(f"[SUCCESS] BTCUSD: ${bid:,.2f} / ${ask:,.2f}")

# Summary
print("\n" + "="*60)
print(" SUMMARY ")
print("="*60)

available_symbols = []
for symbol in ['CRUDEOIL', 'BTCUSD']:
    if symbol in dwx._Market_Data_DB and dwx._Market_Data_DB[symbol]:
        available_symbols.append(symbol)

if available_symbols:
    print(f"Available symbols: {', '.join(available_symbols)}")
    
    if 'CRUDEOIL' in available_symbols:
        print("\n[SUCCESS] WTI Crude Oil confirmed as: CRUDEOIL")
    
    if 'BTCUSD' in available_symbols:
        print("[SUCCESS] Bitcoin confirmed as: BTCUSD")
else:
    print("No symbols returning data at this time")

# Cleanup
print("\nDisconnecting...")
dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_('CRUDEOIL')
dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_('BTCUSD')
dwx._DWX_ZMQ_SHUTDOWN_()
print("Done!")