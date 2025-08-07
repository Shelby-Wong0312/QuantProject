#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Oil/WTI Symbol Variations in MT4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time

print("\n" + "="*60)
print(" WTI Crude Oil Symbol Discovery ")
print("="*60)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Connect
print("\nConnecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='OilTester',
    _verbose=False,
    _poll_timeout=1000
)

time.sleep(2)

# Oil symbol variations to test
oil_symbols = [
    'WTI',
    'USOIL',
    'OIL',
    'CRUDE',
    'CL',
    'WTIUSD',
    'WTI.USD',
    'USOIL.',
    'OIL.WTI',
    'XTIUSD',
    'USOil',
    'US_OIL',
    'WTICrude',
    'CrudeOil',
    'CRUDEOIL',
    'WTI_OIL',
    'BRENT',
    'UKOIL',
    'OIL.BRENT'
]

print(f"\nTesting {len(oil_symbols)} oil symbol variations:")
print("-"*60)

active_oil_symbols = {}

for symbol in oil_symbols:
    print(f"Testing '{symbol}'...", end='', flush=True)
    
    # Subscribe
    dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
    
    # Wait for data
    time.sleep(2)
    
    # Check if we got data
    if symbol in dwx._Market_Data_DB and dwx._Market_Data_DB[symbol]:
        data = dwx._Market_Data_DB[symbol]
        if data:
            latest_timestamp = list(data.keys())[-1]
            bid, ask = data[latest_timestamp]
            
            print(f" [ACTIVE] ${bid:.2f} / ${ask:.2f}")
            active_oil_symbols[symbol] = (bid, ask)
            
            # Show tick count
            tick_count = len(data)
            if tick_count > 1:
                print(f"  Received {tick_count} ticks")
    else:
        print(" [NO DATA]")
    
    # Unsubscribe
    dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)

print("\n" + "="*60)
print(" Results ")
print("="*60)

if active_oil_symbols:
    print(f"\nFound {len(active_oil_symbols)} active oil symbols:\n")
    
    for symbol, (bid, ask) in active_oil_symbols.items():
        spread = ask - bid
        print(f"  {symbol}:")
        print(f"    Bid: ${bid:.2f}")
        print(f"    Ask: ${ask:.2f}")
        print(f"    Spread: ${spread:.2f}")
    
    print("\n[SUCCESS] Oil symbols found!")
    print("\nYou can use these symbols for WTI Crude Oil data:")
    for symbol in active_oil_symbols.keys():
        print(f"  - {symbol}")
else:
    print("\n[WARNING] No oil symbols found")
    print("\nPossible reasons:")
    print("1. Oil trading not available in your MT4")
    print("2. Different symbol naming convention")
    print("3. Market closed or no price feed")

# Save results
if active_oil_symbols:
    with open('oil_symbols.txt', 'w') as f:
        f.write(f"Oil Symbols Found - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        for symbol, (bid, ask) in active_oil_symbols.items():
            f.write(f"{symbol}: ${bid:.2f} / ${ask:.2f}\n")
    print(f"\nResults saved to: oil_symbols.txt")

# Cleanup
print("\nDisconnecting...")
dwx._DWX_ZMQ_SHUTDOWN_()
print("Done!")