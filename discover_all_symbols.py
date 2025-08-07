#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discover all available symbols in MT4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time

print("\n" + "="*60)
print(" MT4 Symbol Discovery Tool ")
print("="*60)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f" Day: {datetime.now().strftime('%A')}")
print("="*60)

# Connect
print("\nConnecting to MT4...")
dwx = DWX_ZeroMQ_Connector(
    _ClientID='SymbolDiscovery',
    _verbose=False,
    _poll_timeout=1000
)

time.sleep(2)

# Common symbol variations to test
symbol_variations = {
    'Bitcoin': ['BTCUSD', 'Bitcoin', 'BTC', 'BTC.USD', 'BTCUSDT', 'XBT', 'XBTUSD'],
    'Ethereum': ['ETHUSD', 'Ethereum', 'ETH', 'ETH.USD', 'ETHUSDT'],
    'Gold': ['XAUUSD', 'GOLD', 'Gold', 'GLD', 'XAU'],
    'Silver': ['XAGUSD', 'SILVER', 'Silver', 'SLV', 'XAG'],
    'EUR/USD': ['EURUSD', 'EUR/USD', 'EUR.USD', 'EURUSD.'],
    'GBP/USD': ['GBPUSD', 'GBP/USD', 'GBP.USD', 'GBPUSD.'],
    'Dow Jones': ['US30', 'DJ30', 'DJI', 'YM', 'US30Cash'],
    'S&P 500': ['SPX500', 'SP500', 'SPX', 'ES', 'US500'],
    'Oil': ['USOIL', 'WTI', 'CL', 'OIL', 'USOIL.'],
    'Ripple': ['XRPUSD', 'XRP', 'Ripple', 'XRP.USD'],
    'Litecoin': ['LTCUSD', 'LTC', 'Litecoin', 'LTC.USD']
}

print("\nTesting symbol variations (2 seconds each):")
print("-"*60)

active_symbols = {}
tested_count = 0

for category, symbols in symbol_variations.items():
    print(f"\n{category}:")
    
    for symbol in symbols:
        tested_count += 1
        print(f"  Testing '{symbol}'...", end='', flush=True)
        
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
                
                # Show price with appropriate precision
                if bid > 1000:
                    print(f" [ACTIVE] ${bid:.2f} / ${ask:.2f}")
                elif bid > 100:
                    print(f" [ACTIVE] {bid:.3f} / {ask:.3f}")
                elif bid > 10:
                    print(f" [ACTIVE] {bid:.4f} / {ask:.4f}")
                else:
                    print(f" [ACTIVE] {bid:.6f} / {ask:.6f}")
                
                active_symbols[symbol] = (category, bid, ask)
                
                # Show recent tick count
                tick_count = len(data)
                if tick_count > 1:
                    print(f"    Received {tick_count} ticks")
        else:
            print(" [NO DATA]")
        
        # Unsubscribe
        dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)

print("\n" + "="*60)
print(" Results ")
print("="*60)

print(f"\nTested {tested_count} symbol variations")

if active_symbols:
    print(f"Found {len(active_symbols)} active symbols:\n")
    
    # Group by category
    categories = {}
    for symbol, (category, bid, ask) in active_symbols.items():
        if category not in categories:
            categories[category] = []
        categories[category].append((symbol, bid, ask))
    
    for category, symbols in categories.items():
        print(f"\n{category}:")
        for symbol, bid, ask in symbols:
            spread = ask - bid
            if bid > 1000:
                print(f"  {symbol}: ${bid:.2f} / ${ask:.2f} (spread: ${spread:.2f})")
            elif bid > 100:
                print(f"  {symbol}: {bid:.3f} / {ask:.3f} (spread: {spread:.3f})")
            elif bid > 10:
                print(f"  {symbol}: {bid:.4f} / {ask:.4f} (spread: {spread:.4f})")
            else:
                print(f"  {symbol}: {bid:.6f} / {ask:.6f} (spread: {spread:.6f})")
    
    print("\n[SUCCESS] Your MT4 connection is working!")
    print("\nYou can use these symbols for data collection:")
    for symbol in active_symbols.keys():
        print(f"  - {symbol}")
    
    # Check for Bitcoin
    btc_found = False
    for symbol in active_symbols:
        if 'BTC' in symbol.upper() or 'BITCOIN' in symbol.upper() or 'XBT' in symbol.upper():
            btc_found = True
            print(f"\n[INFO] Bitcoin is available as: {symbol}")
            break
    
    if not btc_found:
        print("\n[WARNING] Bitcoin not found. Your broker may:")
        print("  - Use a different symbol name")
        print("  - Not offer cryptocurrency trading")
        print("  - Require special account permissions")
else:
    print("[ERROR] No active symbols found")
    print("\nPossible issues:")
    print("1. MT4 is not receiving price feeds")
    print("2. Markets are closed")
    print("3. EA is not configured correctly")
    print("\nPlease check in MT4:")
    print("- Is MT4 connected? (check bottom right corner)")
    print("- Can you see prices updating in Market Watch window?")
    print("- Is the DWX EA running? (smiley face on chart)")

# Save results to file
if active_symbols:
    with open('active_symbols.txt', 'w') as f:
        f.write(f"MT4 Active Symbols - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        for symbol in sorted(active_symbols.keys()):
            category, bid, ask = active_symbols[symbol]
            f.write(f"{symbol} ({category}): {bid} / {ask}\n")
    print(f"\nResults saved to: active_symbols.txt")

# Cleanup
print("\nDisconnecting...")
dwx._DWX_ZMQ_SHUTDOWN_()
print("Done!")