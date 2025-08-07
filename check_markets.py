#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check which markets are sending data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time

print("\n" + "="*60)
print(" Market Availability Check ")
print("="*60)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f" Day: {datetime.now().strftime('%A')}")
print("="*60)

# Connect
print("\nConnecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='MarketChecker',
    _verbose=False,
    _poll_timeout=1000
)

time.sleep(2)

# Test symbols
symbols_to_test = [
    'BTCUSD',   # Bitcoin
    'EURUSD',   # Euro/USD (Forex)
    'GBPUSD',   # GBP/USD (Forex)
    'USDJPY',   # USD/JPY (Forex)
    'XAUUSD',   # Gold
    'XAGUSD',   # Silver
    'US30',     # Dow Jones
    'SPX500',   # S&P 500
    'USOIL',    # Oil
]

print("\nTesting symbols (waiting 5 seconds each):")
print("-"*60)

active_symbols = []

for symbol in symbols_to_test:
    print(f"\nTesting {symbol}...")
    
    # Subscribe
    dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
    
    # Wait for data
    time.sleep(5)
    
    # Check if we got data
    if symbol in dwx._Market_Data_DB and dwx._Market_Data_DB[symbol]:
        data = dwx._Market_Data_DB[symbol]
        latest_timestamp = list(data.keys())[-1]
        bid, ask = data[latest_timestamp]
        
        # Format based on symbol type
        if 'JPY' in symbol:
            print(f"  [ACTIVE]: {bid:.3f} / {ask:.3f}")
        elif 'XAU' in symbol or 'XAG' in symbol or 'BTC' in symbol or 'US30' in symbol or 'SPX' in symbol:
            print(f"  [ACTIVE]: {bid:.2f} / {ask:.2f}")
        else:
            print(f"  [ACTIVE]: {bid:.5f} / {ask:.5f}")
        
        active_symbols.append(symbol)
        
        # Show tick frequency
        tick_count = len(data)
        print(f"  Ticks received: {tick_count}")
    else:
        print(f"  [NO DATA]")
    
    # Unsubscribe
    dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)

print("\n" + "="*60)
print(" Summary ")
print("="*60)

if active_symbols:
    print(f"Active symbols: {', '.join(active_symbols)}")
    print(f"Total active: {len(active_symbols)}/{len(symbols_to_test)}")
    
    print("\n[SUCCESS] Your MT4 connection is working!")
    print("[SUCCESS] You can collect data from these symbols")
    
    if 'BTCUSD' not in active_symbols:
        print("\n[WARNING] BTCUSD is not sending data. Possible reasons:")
        print("  1. Symbol name might be different (try Bitcoin, BTC.USD, etc.)")
        print("  2. Crypto trading might not be available")
        print("  3. Market might be closed for maintenance")
else:
    print("[ERROR] No symbols are sending data")
    print("\nPossible issues:")
    print("1. Markets are closed (weekend/holiday)")
    print("2. MT4 is not receiving price feeds")
    print("3. EA configuration issue")
    print("\nPlease check:")
    print("- Is MT4 connected to server? (bottom right corner)")
    print("- Can you see prices updating in Market Watch?")

# Cleanup
print("\nDisconnecting...")
dwx._DWX_ZMQ_SHUTDOWN_()
print("Done!")