#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Market Status Report
Testing BTCUSD and CRUDEOIL availability
"""

import zmq
import time
from datetime import datetime
import pandas as pd

print("\n" + "="*70)
print(" Market Status Report ")
print("="*70)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}")
print("="*70)

context = zmq.Context()
push = context.socket(zmq.PUSH)
push.connect("tcp://localhost:32768")
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:32770")
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.setsockopt(zmq.RCVTIMEO, 100)

# Test all requested symbols
test_symbols = {
    'BTCUSD': 'Bitcoin',
    'CRUDEOIL': 'WTI Crude Oil',
    'XRPUSD': 'Ripple',
    'GOLD': 'Gold',
    'ETHUSD': 'Ethereum',
    'AUDUSD': 'AUD/USD Forex'
}

print("\n[1] Testing Symbol Availability:")
print("-"*70)

active_symbols = {}
inactive_symbols = []

for symbol, name in test_symbols.items():
    print(f"  {symbol:10} ({name})...", end='', flush=True)
    
    # Subscribe
    push.send_string(f"TRACK_PRICES;{symbol}")
    time.sleep(0.5)
    
    # Check for data
    found = False
    for _ in range(10):
        try:
            msg = sub.recv_string()
            if symbol in msg and ":|:" in msg:
                parts = msg.split(":|:")
                if len(parts) >= 2:
                    values = parts[1].split(";")
                    if len(values) >= 2:
                        try:
                            bid = float(values[0])
                            ask = float(values[1])
                            
                            if bid > 10000:
                                print(f" [ACTIVE] ${bid:,.2f}")
                            elif bid > 1000:
                                print(f" [ACTIVE] ${bid:.2f}")
                            elif bid < 10:
                                print(f" [ACTIVE] {bid:.6f}")
                            else:
                                print(f" [ACTIVE] {bid:.2f}")
                            
                            active_symbols[symbol] = {
                                'name': name,
                                'bid': bid,
                                'ask': ask
                            }
                            found = True
                            break
                        except ValueError:
                            pass
        except zmq.Again:
            pass
    
    if not found:
        print(" [NOT ACTIVE]")
        inactive_symbols.append(symbol)

# Report
print("\n[2] Status Summary:")
print("-"*70)

if 'BTCUSD' in active_symbols:
    print(f"[SUCCESS] BITCOIN (BTCUSD): AVAILABLE")
    print(f"  Current price: ${active_symbols['BTCUSD']['bid']:,.2f}")
elif 'BTCUSD' in inactive_symbols:
    print(f"[FAILED] BITCOIN (BTCUSD): NOT ACTIVE")
    print(f"  The symbol exists but is not sending data currently")

if 'CRUDEOIL' in active_symbols:
    print(f"\n[SUCCESS] WTI CRUDE OIL (CRUDEOIL): AVAILABLE")
    print(f"  Current price: ${active_symbols['CRUDEOIL']['bid']:.2f}")
elif 'CRUDEOIL' in inactive_symbols:
    print(f"\n[FAILED] WTI CRUDE OIL (CRUDEOIL): NOT ACTIVE")
    print(f"  The symbol may exist but is not sending data")
    print(f"  Possible reasons:")
    print(f"  - Market closed (check trading hours)")
    print(f"  - Account permissions needed")
    print(f"  - Symbol not available from broker")

# Collect brief data from active symbols
if active_symbols:
    print("\n[3] Collecting 5 seconds of data from active symbols...")
    print("-"*70)
    
    data_collected = {sym: [] for sym in active_symbols}
    start = time.time()
    
    while time.time() - start < 5:
        try:
            msg = sub.recv_string()
            if ":|:" in msg:
                parts = msg.split(":|:")
                symbol = parts[0]
                
                if symbol in active_symbols:
                    values = parts[1].split(";")
                    if len(values) >= 2:
                        try:
                            bid = float(values[0])
                            ask = float(values[1])
                            data_collected[symbol].append({
                                'time': datetime.now(),
                                'bid': bid,
                                'ask': ask
                            })
                        except ValueError:
                            pass
        except zmq.Again:
            pass
    
    for symbol, ticks in data_collected.items():
        if ticks:
            print(f"  {symbol}: {len(ticks)} ticks collected")

# Final recommendations
print("\n[4] Recommendations:")
print("-"*70)

if 'BTCUSD' in active_symbols:
    print("[OK] Bitcoin trading is available - you can collect BTCUSD data")
else:
    print("[WARNING] Bitcoin (BTCUSD) is not currently active")
    print("  - Check if crypto markets are open")
    print("  - Verify MT4 is connected to server")

if 'CRUDEOIL' not in active_symbols:
    print("\n[WARNING] WTI Crude Oil (CRUDEOIL) is not currently active")
    print("  Alternatives available:")
    if 'GOLD' in active_symbols:
        print("  - GOLD: Commodity trading available")
    if active_symbols:
        print(f"  - Active symbols: {', '.join(active_symbols.keys())}")
    
    print("\n  To enable CRUDEOIL:")
    print("  1. Check MT4 Market Watch window")
    print("  2. Right-click and select 'Show All'")
    print("  3. Look for CRUDEOIL or similar oil symbols")
    print("  4. Verify trading hours (usually Sun 6pm - Fri 5pm ET)")
    print("  5. Contact broker if symbol not available")

# Save report
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'market_status_{timestamp}.txt', 'w') as f:
    f.write("Market Status Report\n")
    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}\n")
    f.write("="*60 + "\n\n")
    
    f.write("Active Symbols:\n")
    for symbol, info in active_symbols.items():
        f.write(f"  - {symbol}: {info['name']} (${info['bid']})\n")
    
    f.write("\nInactive Symbols:\n")
    for symbol in inactive_symbols:
        f.write(f"  - {symbol}\n")
    
    f.write("\nBTC Status: ")
    f.write("AVAILABLE\n" if 'BTCUSD' in active_symbols else "NOT ACTIVE\n")
    
    f.write("WTI Status: ")
    f.write("AVAILABLE\n" if 'CRUDEOIL' in active_symbols else "NOT ACTIVE\n")

print(f"\n[5] Report saved to: market_status_{timestamp}.txt")

# Cleanup
push.close()
sub.close()
context.term()

print("\n" + "="*70)
print(" Complete ")
print("="*70)