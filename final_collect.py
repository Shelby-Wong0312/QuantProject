#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final Collection Script - BTCUSD & Available Markets
"""

import zmq
import time
from datetime import datetime
import pandas as pd

print("\n" + "="*70)
print(" Market Data Collection - Direct ZeroMQ ")
print("="*70)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

context = zmq.Context()

# Setup sockets
push = context.socket(zmq.PUSH)
push.connect("tcp://localhost:32768")

sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:32770")
sub.setsockopt_string(zmq.SUBSCRIBE, "")

print("\n[INFO] Testing available symbols...")

# Test symbols we found earlier
test_symbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'GOLD', 'XAGUSD', 'AUDUSD']
active_symbols = []

for symbol in test_symbols:
    print(f"  {symbol}...", end='', flush=True)
    
    # Subscribe
    cmd = f"TRACK_PRICES;{symbol}"
    push.send_string(cmd)
    time.sleep(0.5)
    
    # Check for data
    received = False
    try:
        sub.setsockopt(zmq.RCVTIMEO, 1000)
        for _ in range(3):
            msg = sub.recv_string()
            if symbol in msg:
                received = True
                parts = msg.split(":|:")
                if len(parts) >= 2:
                    values = parts[1].split(";")
                    if len(values) >= 2:
                        bid = float(values[0])
                        if bid > 10000:
                            print(f" [ACTIVE] ${bid:.2f}")
                        elif bid > 1000:
                            print(f" [ACTIVE] ${bid:.2f}")
                        else:
                            print(f" [ACTIVE] {bid:.6f}")
                        active_symbols.append(symbol)
                        break
    except zmq.Again:
        pass
    
    if not received:
        print(" [NO DATA]")

print(f"\n[INFO] Active symbols: {', '.join(active_symbols)}")

# WTI Test
print("\n[INFO] Testing WTI Crude Oil...")
wti_symbols = ['WTI', 'WTIUSD', 'USOIL', 'OIL', 'XTIUSD', 'CRUDE']
wti_found = None

for symbol in wti_symbols:
    print(f"  {symbol}...", end='', flush=True)
    
    cmd = f"TRACK_PRICES;{symbol}"
    push.send_string(cmd)
    time.sleep(0.5)
    
    received = False
    try:
        sub.setsockopt(zmq.RCVTIMEO, 500)
        for _ in range(2):
            msg = sub.recv_string()
            if symbol in msg:
                received = True
                wti_found = symbol
                print(" [FOUND]")
                active_symbols.append(symbol)
                break
    except zmq.Again:
        pass
    
    if not received:
        print(" [NO DATA]")
    
    if wti_found:
        break

if wti_found:
    print(f"\n[SUCCESS] WTI Crude Oil available as: {wti_found}")
else:
    print("\n[INFO] WTI Crude Oil not available in your MT4")

# Collect data
if active_symbols:
    print(f"\n[INFO] Collecting data for 20 seconds...")
    print("="*70)
    
    data = {symbol: [] for symbol in active_symbols}
    start_time = time.time()
    last_display = start_time
    
    sub.setsockopt(zmq.RCVTIMEO, 100)
    
    while time.time() - start_time < 20:
        try:
            msg = sub.recv_string()
            if ":|:" in msg:
                parts = msg.split(":|:")
                symbol = parts[0]
                
                if symbol in active_symbols:
                    values = parts[1].split(";")
                    if len(values) >= 2:
                        bid = float(values[0])
                        ask = float(values[1])
                        
                        data[symbol].append({
                            'timestamp': datetime.now(),
                            'bid': bid,
                            'ask': ask,
                            'spread': ask - bid
                        })
        except zmq.Again:
            pass
        
        # Display every 5 seconds
        if time.time() - last_display >= 5:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Ticks collected:")
            for sym in active_symbols:
                if data[sym]:
                    latest = data[sym][-1]
                    count = len(data[sym])
                    
                    if sym == 'BTCUSD':
                        print(f"  BTC: ${latest['bid']:,.2f} ({count} ticks)")
                    elif sym == 'ETHUSD':
                        print(f"  ETH: ${latest['bid']:.2f} ({count} ticks)")
                    elif sym == 'XRPUSD':
                        print(f"  XRP: ${latest['bid']:.6f} ({count} ticks)")
                    elif sym == 'GOLD':
                        print(f"  Gold: ${latest['bid']:,.2f} ({count} ticks)")
                    elif sym == 'XAGUSD':
                        print(f"  Silver: ${latest['bid']:.2f} ({count} ticks)")
                    elif sym == 'AUDUSD':
                        print(f"  AUD/USD: {latest['bid']:.6f} ({count} ticks)")
                    elif wti_found and sym == wti_found:
                        print(f"  WTI Oil: ${latest['bid']:.2f} ({count} ticks)")
            
            last_display = time.time()
    
    print("\n" + "="*70)
    print("[INFO] Collection complete")
    
    # Save data
    print("\n[INFO] Saving data...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for symbol in active_symbols:
        if data[symbol]:
            df = pd.DataFrame(data[symbol])
            filename = f"{symbol}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"  Saved {len(df)} ticks to {filename}")
    
    # Summary
    print("\n[SUMMARY]")
    for symbol in active_symbols:
        if data[symbol]:
            df = pd.DataFrame(data[symbol])
            print(f"\n{symbol}:")
            print(f"  Ticks: {len(df)}")
            print(f"  Bid range: {df['bid'].min():.6f} - {df['bid'].max():.6f}")
            print(f"  Avg spread: {df['spread'].mean():.6f}")
    
    # Bitcoin status
    if 'BTCUSD' in active_symbols and data['BTCUSD']:
        print(f"\n[SUCCESS] Bitcoin (BTCUSD) data collected successfully!")
    else:
        print(f"\n[INFO] Bitcoin (BTCUSD) not available at this time")
    
    # WTI status  
    if wti_found:
        print(f"[SUCCESS] WTI Crude Oil ({wti_found}) data collected successfully!")
    else:
        print(f"[INFO] WTI Crude Oil not available in your MT4 broker")
        print("       Alternatives: Gold and Silver commodities are available")

else:
    print("\n[ERROR] No active symbols found")

# Cleanup
push.close()
sub.close()
context.term()

print("\n" + "="*70)
print(" Complete ")
print("="*70)