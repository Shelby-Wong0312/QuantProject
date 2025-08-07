#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test available symbols in MT4"""

import zmq
import time
from datetime import datetime

print("\n" + "="*50)
print(" MT4 Symbol Discovery ")
print("="*50)
print(f" Time: {datetime.now().strftime('%H:%M:%S')}")

context = zmq.Context()

# Function to send command
def send_command(command):
    push = context.socket(zmq.PUSH)
    push.connect("tcp://localhost:32768")
    push.send_string(command)
    push.close()

# Function to check for data
def check_data(timeout=3000):
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://localhost:32770")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, timeout)
    
    received = []
    try:
        for i in range(10):
            msg = sub.recv_string()
            if ":|:" in msg:
                symbol = msg.split(":|:")[0]
                data = msg.split(":|:")[1]
                received.append((symbol, data))
    except zmq.Again:
        pass
    
    sub.close()
    return received

# Test forex symbols
print("\n1. Testing Forex Symbols:")
forex = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
for symbol in forex:
    print(f"   Testing {symbol}...", end='')
    
    # Use DWX format for subscription
    cmd = f"TRACK_PRICES;{symbol}"
    send_command(cmd)
    time.sleep(0.5)
    
    data = check_data(500)
    if data:
        print(f" [ACTIVE] - {data[0][1][:30]}...")
    else:
        print(" [NO DATA]")

# Test indices
print("\n2. Testing Indices:")
indices = ['US30', 'SPX500', 'NAS100', 'DAX30', 'FTSE100']
for symbol in indices:
    print(f"   Testing {symbol}...", end='')
    
    cmd = f"TRACK_PRICES;{symbol}"
    send_command(cmd)
    time.sleep(0.5)
    
    data = check_data(500)
    if data:
        print(f" [ACTIVE] - {data[0][1][:30]}...")
    else:
        print(" [NO DATA]")

# Test commodities
print("\n3. Testing Commodities:")
commodities = ['XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'OIL', 'USOIL']
for symbol in commodities:
    print(f"   Testing {symbol}...", end='')
    
    cmd = f"TRACK_PRICES;{symbol}"
    send_command(cmd)
    time.sleep(0.5)
    
    data = check_data(500)
    if data:
        print(f" [ACTIVE] - {data[0][1][:30]}...")
    else:
        print(" [NO DATA]")

# Test crypto
print("\n4. Testing Crypto:")
crypto = ['BTCUSD', 'Bitcoin', 'ETHUSD', 'Ethereum', 'LTCUSD', 'XRPUSD']
for symbol in crypto:
    print(f"   Testing {symbol}...", end='')
    
    cmd = f"TRACK_PRICES;{symbol}"
    send_command(cmd)
    time.sleep(0.5)
    
    data = check_data(500)
    if data:
        print(f" [ACTIVE] - {data[0][1][:30]}...")
    else:
        print(" [NO DATA]")

# Collect all active symbols
print("\n5. Collecting all active data for 3 seconds...")
time.sleep(3)

all_data = check_data(3000)
symbols_found = set()
for symbol, data in all_data:
    symbols_found.add(symbol)
    values = data.split(";")
    if len(values) >= 2:
        print(f"   {symbol}: {values[0]}/{values[1]}")

print("\n" + "="*50)
print(" Summary ")
print("="*50)
if symbols_found:
    print(f"Active symbols: {', '.join(sorted(symbols_found))}")
    print(f"Total: {len(symbols_found)} symbols")
else:
    print("No active symbols found")
    print("Please check:")
    print("1. Is market open?")
    print("2. Are symbols available in MT4?")
    print("3. Is EA configured correctly?")

context.term()