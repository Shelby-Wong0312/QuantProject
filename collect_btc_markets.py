#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BTC and Market Data Collection
Confirmed working symbols: BTCUSD, ETHUSD, XRPUSD, GOLD, AUDUSD
WTI Status: Not available in this MT4 broker
"""

import zmq
import time
from datetime import datetime
import pandas as pd
import json

def main():
    print("\n" + "="*70)
    print(" BTC & Market Data Collection ")
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
    sub.setsockopt(zmq.RCVTIMEO, 100)
    
    # Working symbols
    symbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'GOLD', 'AUDUSD']
    
    print("\n[INFO] Subscribing to symbols...")
    for symbol in symbols:
        cmd = f"TRACK_PRICES;{symbol}"
        push.send_string(cmd)
        print(f"  - {symbol}")
        time.sleep(0.2)
    
    print("\n[INFO] WTI Crude Oil Status:")
    print("  [NOT AVAILABLE] WTI is not offered by your MT4 broker")
    print("  Alternative: Using GOLD as commodity benchmark")
    
    # Collect data
    print(f"\n[INFO] Collecting data for 30 seconds...")
    print("="*70)
    
    data = {symbol: [] for symbol in symbols}
    start_time = time.time()
    last_display = start_time
    
    while time.time() - start_time < 30:
        try:
            msg = sub.recv_string()
            if ":|:" in msg:
                parts = msg.split(":|:")
                symbol = parts[0]
                
                if symbol in symbols:
                    values = parts[1].split(";")
                    if len(values) >= 2:
                        try:
                            bid = float(values[0])
                            ask = float(values[1])
                            
                            data[symbol].append({
                                'timestamp': datetime.now(),
                                'bid': bid,
                                'ask': ask,
                                'spread': ask - bid
                            })
                        except ValueError:
                            pass
        except zmq.Again:
            pass
        
        # Display every 5 seconds
        if time.time() - last_display >= 5:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Live Prices:")
            
            if data['BTCUSD']:
                latest = data['BTCUSD'][-1]
                print(f"  BTC/USD:  ${latest['bid']:,.2f} / ${latest['ask']:,.2f} | Spread: ${latest['spread']:.2f} | Ticks: {len(data['BTCUSD'])}")
            
            if data['ETHUSD']:
                latest = data['ETHUSD'][-1]
                print(f"  ETH/USD:  ${latest['bid']:,.2f} / ${latest['ask']:,.2f} | Spread: ${latest['spread']:.2f} | Ticks: {len(data['ETHUSD'])}")
            
            if data['XRPUSD']:
                latest = data['XRPUSD'][-1]
                print(f"  XRP/USD:  ${latest['bid']:.6f} / ${latest['ask']:.6f} | Spread: ${latest['spread']:.6f} | Ticks: {len(data['XRPUSD'])}")
            
            if data['GOLD']:
                latest = data['GOLD'][-1]
                print(f"  Gold:     ${latest['bid']:,.2f} / ${latest['ask']:,.2f} | Spread: ${latest['spread']:.2f} | Ticks: {len(data['GOLD'])}")
            
            if data['AUDUSD']:
                latest = data['AUDUSD'][-1]
                print(f"  AUD/USD:  {latest['bid']:.6f} / {latest['ask']:.6f} | Spread: {latest['spread']:.6f} | Ticks: {len(data['AUDUSD'])}")
            
            last_display = time.time()
        
        time.sleep(0.05)
    
    print("\n" + "="*70)
    print("[INFO] Collection complete")
    
    # Analysis
    print("\n[ANALYSIS]")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = []
    
    for symbol in symbols:
        if data[symbol]:
            df = pd.DataFrame(data[symbol])
            
            print(f"\n{symbol}:")
            print(f"  Total ticks: {len(df)}")
            print(f"  Time range: {df['timestamp'].min().strftime('%H:%M:%S')} - {df['timestamp'].max().strftime('%H:%M:%S')}")
            
            if symbol == 'BTCUSD':
                print(f"  Price range: ${df['bid'].min():,.2f} - ${df['bid'].max():,.2f}")
                print(f"  Current: ${df['bid'].iloc[-1]:,.2f}")
                print(f"  Average spread: ${df['spread'].mean():.2f}")
            elif symbol in ['ETHUSD', 'GOLD']:
                print(f"  Price range: ${df['bid'].min():.2f} - ${df['bid'].max():.2f}")
                print(f"  Current: ${df['bid'].iloc[-1]:.2f}")
                print(f"  Average spread: ${df['spread'].mean():.2f}")
            else:
                print(f"  Price range: {df['bid'].min():.6f} - {df['bid'].max():.6f}")
                print(f"  Current: {df['bid'].iloc[-1]:.6f}")
                print(f"  Average spread: {df['spread'].mean():.6f}")
            
            # Calculate volatility
            if len(df) > 1:
                returns = df['bid'].pct_change().dropna()
                volatility = returns.std() * 100
                print(f"  Volatility: {volatility:.4f}%")
            
            # Save data
            filename = f"{symbol}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            saved_files.append(filename)
    
    print(f"\n[INFO] Data saved to {len(saved_files)} files")
    
    # Final summary
    print("\n" + "="*70)
    print(" SUMMARY ")
    print("="*70)
    
    if data['BTCUSD']:
        btc_df = pd.DataFrame(data['BTCUSD'])
        print(f"\n[SUCCESS] Bitcoin (BTCUSD) collected successfully!")
        print(f"  - {len(btc_df)} ticks collected")
        print(f"  - Current price: ${btc_df['bid'].iloc[-1]:,.2f}")
        print(f"  - 30-second change: ${btc_df['bid'].iloc[-1] - btc_df['bid'].iloc[0]:+,.2f}")
    
    print(f"\n[INFO] WTI Crude Oil:")
    print(f"  - Status: NOT AVAILABLE in your MT4")
    print(f"  - Alternative: Gold commodity data collected")
    print(f"  - Suggestion: Contact broker for oil trading access")
    
    print(f"\n[INFO] Other markets collected:")
    print(f"  - Ethereum (ETH): Cryptocurrency")
    print(f"  - Ripple (XRP): Cryptocurrency")
    print(f"  - Gold: Commodity (substitute for WTI)")
    print(f"  - AUD/USD: Forex pair")
    
    # Cleanup
    push.close()
    sub.close()
    context.term()
    
    print("\n" + "="*70)
    print(" Complete ")
    print("="*70)

if __name__ == "__main__":
    main()