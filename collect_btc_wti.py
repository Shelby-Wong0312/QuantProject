#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect BTC and search for WTI Crude Oil
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import pandas as pd

class BTCWTICollector:
    def __init__(self):
        self.dwx = None
        self.data = {
            'BTCUSD': [],
            'ETHUSD': [],
            'XRPUSD': [],
            'GOLD': [],
            'XAGUSD': [],
            'AUDUSD': []
        }
        self.active_symbols = []
        
    def connect(self):
        """Connect to MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='BTCWTICollector',
            _verbose=False,
            _poll_timeout=1000
        )
        
        time.sleep(3)  # Longer wait for connection
        
        # Get account info
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Getting account info...")
        self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        # Display account info
        if hasattr(self.dwx, 'account_info_DB') and self.dwx.account_info_DB:
            for acc_num, info_list in self.dwx.account_info_DB.items():
                if info_list:
                    account_info = info_list[0]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Account: {acc_num}")
                    print(f"  Balance: ${account_info.get('account_balance', 0):.2f}")
        
        return True
    
    def test_wti_symbols(self):
        """Test all possible WTI symbol variations"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing WTI Crude Oil variations...")
        
        wti_variations = [
            # Standard WTI names
            'WTI', 'WTIUSD', 'WTI.USD', 'WTI/USD',
            # US Oil variations
            'USOIL', 'USOil', 'US.OIL', 'US_OIL', 'USOIL.', 'USOil.Cash',
            # XTI variations
            'XTIUSD', 'XTI', 'XTI.USD', 'XTI/USD',
            # Generic oil
            'OIL', 'OIL.WTI', 'OIL_WTI', 'OILWTI',
            # CL futures
            'CL', 'CLZ24', 'CL!', 'CL.F',
            # Crude variations
            'CRUDE', 'CRUDEOIL', 'CrudeOil', 'CRUDE.OIL',
            # Light crude
            'LIGHT.CRUDE', 'LIGHTCRUDE', 'WTI.CRUDE',
            # Cash variations
            'OIL.Cash', 'WTI.Cash', 'CRUDE.Cash',
            # Symbol with prefix/suffix
            '#WTI', '$WTI', 'WTI#', 'WTI$',
            # Brent (for comparison)
            'BRENT', 'UKOIL', 'BRENT.OIL'
        ]
        
        wti_found = None
        
        for symbol in wti_variations:
            print(f"  Testing '{symbol}'...", end='', flush=True)
            
            # Subscribe with longer wait
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(2.5)  # Longer wait for data
            
            # Check for data
            if symbol in self.dwx._Market_Data_DB and self.dwx._Market_Data_DB[symbol]:
                data = self.dwx._Market_Data_DB[symbol]
                if data and len(data) > 0:
                    latest_timestamp = list(data.keys())[-1]
                    bid, ask = data[latest_timestamp]
                    
                    print(f" [FOUND] ${bid:.2f} / ${ask:.2f}")
                    wti_found = symbol
                    
                    # Add to data collection
                    self.data[symbol] = []
                    
                    # Don't unsubscribe yet, keep it for collection
                    break
            else:
                print(" [NOT AVAILABLE]")
            
            # Unsubscribe if not found
            self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
        
        if wti_found:
            print(f"\n[SUCCESS] WTI Crude Oil found as: {wti_found}")
            self.active_symbols.append(wti_found)
        else:
            print(f"\n[INFO] WTI Crude Oil not available")
            print("       Your MT4 broker may not offer oil trading")
        
        return wti_found
    
    def subscribe_all(self):
        """Subscribe to BTC and other symbols"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Subscribing to available symbols...")
        
        # Test and subscribe to each symbol
        test_symbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'GOLD', 'XAGUSD', 'AUDUSD']
        
        for symbol in test_symbols:
            print(f"  Testing {symbol}...", end='', flush=True)
            
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(2.5)  # Longer wait
            
            # Check if getting data
            if symbol in self.dwx._Market_Data_DB and self.dwx._Market_Data_DB[symbol]:
                data = self.dwx._Market_Data_DB[symbol]
                if data and len(data) > 0:
                    latest_timestamp = list(data.keys())[-1]
                    bid, ask = data[latest_timestamp]
                    
                    if bid > 10000:  # BTC
                        print(f" [ACTIVE] ${bid:.2f}")
                    elif bid > 1000:  # Gold
                        print(f" [ACTIVE] ${bid:.2f}")
                    elif bid < 10:  # Crypto/Forex
                        print(f" [ACTIVE] {bid:.6f}")
                    else:
                        print(f" [ACTIVE] {bid:.4f}")
                    
                    self.active_symbols.append(symbol)
                else:
                    print(" [NO DATA]")
                    self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
            else:
                print(" [NO DATA]")
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Active symbols: {', '.join(self.active_symbols)}")
    
    def collect_data(self, duration_seconds=30):
        """Collect data"""
        if not self.active_symbols:
            print("\n[ERROR] No active symbols to collect")
            return {}
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collecting data for {duration_seconds} seconds...")
        print("="*70)
        
        start_time = time.time()
        last_display = start_time
        tick_counts = {symbol: 0 for symbol in self.active_symbols}
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Check all active symbols
            for symbol in self.active_symbols:
                if symbol in self.dwx._Market_Data_DB:
                    data = self.dwx._Market_Data_DB[symbol]
                    if data:
                        # Get new ticks
                        for timestamp, (bid, ask) in list(data.items())[-5:]:  # Last 5 ticks
                            # Check if this is a new tick
                            if not any(t['timestamp'] == timestamp for t in self.data[symbol]):
                                tick_data = {
                                    'timestamp': timestamp,
                                    'bid': bid,
                                    'ask': ask,
                                    'spread': ask - bid
                                }
                                self.data[symbol].append(tick_data)
                                tick_counts[symbol] += 1
            
            # Display every 2 seconds
            if current_time - last_display >= 2:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current prices:")
                
                for symbol in self.active_symbols:
                    if self.data[symbol]:
                        latest = self.data[symbol][-1]
                        bid = latest['bid']
                        ask = latest['ask']
                        
                        if symbol == 'BTCUSD':
                            print(f"  BTC/USD:  ${bid:,.2f} / ${ask:,.2f} | Ticks: {tick_counts[symbol]}")
                        elif symbol == 'ETHUSD':
                            print(f"  ETH/USD:  ${bid:,.2f} / ${ask:,.2f} | Ticks: {tick_counts[symbol]}")
                        elif symbol == 'XRPUSD':
                            print(f"  XRP/USD:  ${bid:.6f} / ${ask:.6f} | Ticks: {tick_counts[symbol]}")
                        elif symbol == 'GOLD':
                            print(f"  Gold:     ${bid:,.2f} / ${ask:,.2f} | Ticks: {tick_counts[symbol]}")
                        elif symbol == 'XAGUSD':
                            print(f"  Silver:   ${bid:,.2f} / ${ask:,.2f} | Ticks: {tick_counts[symbol]}")
                        elif symbol == 'AUDUSD':
                            print(f"  AUD/USD:  {bid:.6f} / {ask:.6f} | Ticks: {tick_counts[symbol]}")
                        elif 'WTI' in symbol.upper() or 'OIL' in symbol.upper():
                            print(f"  WTI Oil:  ${bid:.2f} / ${ask:.2f} | Ticks: {tick_counts[symbol]}")
                        else:
                            print(f"  {symbol}:  {bid} / {ask} | Ticks: {tick_counts[symbol]}")
                
                last_display = current_time
            
            time.sleep(0.1)
            
            # Progress update
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed - int(elapsed) < 0.1:
                total_ticks = sum(tick_counts.values())
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: {int(elapsed)}/{duration_seconds}s | Total ticks: {total_ticks}")
        
        print("="*70)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collection complete")
        
        # Summary
        print("\nSummary:")
        for symbol in self.active_symbols:
            if tick_counts[symbol] > 0:
                print(f"  {symbol}: {tick_counts[symbol]} ticks collected")
        
        # Check BTC status
        if 'BTCUSD' in self.active_symbols and tick_counts['BTCUSD'] > 0:
            btc_data = self.data['BTCUSD']
            prices = [t['bid'] for t in btc_data]
            print(f"\n[SUCCESS] Bitcoin data collected!")
            print(f"  Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
        
        return tick_counts
    
    def save_data(self):
        """Save collected data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        for symbol in self.active_symbols:
            if self.data[symbol]:
                df = pd.DataFrame(self.data[symbol])
                filename = f"{symbol}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                saved_files.append(filename)
                print(f"  Saved {len(df)} ticks to {filename}")
        
        return saved_files
    
    def disconnect(self):
        """Disconnect"""
        if self.dwx:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            
            for symbol in self.active_symbols:
                try:
                    self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
                except:
                    pass
            
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Disconnected")

def main():
    print("\n" + "="*70)
    print(" BTC & WTI Crude Oil Data Collection ")
    print("="*70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    collector = BTCWTICollector()
    
    try:
        if collector.connect():
            # First test for WTI
            wti_symbol = collector.test_wti_symbols()
            
            # Then subscribe to other symbols
            collector.subscribe_all()
            
            if collector.active_symbols:
                # Collect data
                print("\nPress Ctrl+C to stop early")
                tick_counts = collector.collect_data(duration_seconds=30)
                
                # Save data
                if any(tick_counts.values()):
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving data...")
                    collector.save_data()
            else:
                print("\n[ERROR] No active symbols found")
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        collector.disconnect()
    
    print("\n" + "="*70)
    print(" Complete ")
    print("="*70)

if __name__ == "__main__":
    main()