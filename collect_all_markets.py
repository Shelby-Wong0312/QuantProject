#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect All Available Market Data from MT4
Including attempt to find WTI Crude Oil and other commodities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import pandas as pd
import json

class MarketDataCollector:
    def __init__(self):
        self.dwx = None
        self.market_data = {}
        self.last_prices = {}
        self.active_symbols = []
        
    def connect(self):
        """Connect to MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='MarketCollector',
            _verbose=False,
            _poll_timeout=1000
        )
        
        time.sleep(2)
        
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
                    print(f"  Equity: ${account_info.get('account_equity', 0):.2f}")
        
        return True
    
    def discover_symbols(self):
        """Discover available symbols"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Discovering available symbols...")
        
        # Test symbols including WTI attempts
        test_symbols = {
            # Known working
            'XRPUSD': 'Ripple (XRP)',
            'Ethereum': 'Ethereum (ETH)',
            'LTCUSD': 'Litecoin (LTC)',
            'GOLD': 'Gold',
            
            # WTI Crude Oil attempts
            'WTI': 'WTI Crude Oil',
            'WTIUSD': 'WTI Crude Oil',
            'USOIL': 'US Oil',
            'OIL': 'Oil',
            'XTIUSD': 'WTI (XTI)',
            'CL': 'Crude Light',
            
            # Other commodities
            'SILVER': 'Silver',
            'COPPER': 'Copper',
            'NATGAS': 'Natural Gas',
            'WHEAT': 'Wheat',
            'CORN': 'Corn',
            'SUGAR': 'Sugar',
            'COFFEE': 'Coffee',
            
            # Forex majors
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'USDJPY': 'USD/JPY',
            'AUDUSD': 'AUD/USD',
            
            # Indices
            'US30': 'Dow Jones',
            'SPX500': 'S&P 500',
            'NAS100': 'NASDAQ',
            'DAX30': 'DAX',
            
            # More crypto attempts
            'Bitcoin': 'Bitcoin',
            'BTCUSD': 'Bitcoin',
            'BTC': 'Bitcoin'
        }
        
        found_symbols = {}
        
        for symbol, name in test_symbols.items():
            print(f"  Testing {symbol} ({name})...", end='', flush=True)
            
            # Subscribe
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(1)
            
            # Check for data
            if symbol in self.dwx._Market_Data_DB and self.dwx._Market_Data_DB[symbol]:
                data = self.dwx._Market_Data_DB[symbol]
                if data:
                    latest_timestamp = list(data.keys())[-1]
                    bid, ask = data[latest_timestamp]
                    
                    if bid > 1000:
                        print(f" [FOUND] ${bid:.2f}")
                    elif bid > 100:
                        print(f" [FOUND] {bid:.3f}")
                    else:
                        print(f" [FOUND] {bid:.6f}")
                    
                    found_symbols[symbol] = {
                        'name': name,
                        'bid': bid,
                        'ask': ask
                    }
            else:
                print(" [NOT AVAILABLE]")
            
            # Unsubscribe
            self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(found_symbols)} active symbols")
        
        # Set active symbols
        self.active_symbols = list(found_symbols.keys())
        
        # Initialize data storage
        for symbol in self.active_symbols:
            self.market_data[symbol] = []
        
        return found_symbols
    
    def subscribe_all(self):
        """Subscribe to all discovered symbols"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Subscribing to {len(self.active_symbols)} symbols...")
        
        for symbol in self.active_symbols:
            print(f"  Subscribing to {symbol}...")
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(0.3)
    
    def collect_data(self, duration_seconds=30):
        """Collect data for specified duration"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting data collection for {duration_seconds} seconds...")
        print("="*70)
        
        start_time = time.time()
        last_display_time = start_time
        tick_counts = {symbol: 0 for symbol in self.active_symbols}
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Check market data
            if self.dwx._Market_Data_DB:
                for symbol, data in self.dwx._Market_Data_DB.items():
                    if data and symbol in self.active_symbols:
                        # Get latest tick
                        timestamps = list(data.keys())
                        if timestamps:
                            latest_timestamp = timestamps[-1]
                            bid, ask = data[latest_timestamp]
                            
                            # Save data
                            tick_data = {
                                'timestamp': datetime.now(),
                                'bid': bid,
                                'ask': ask,
                                'spread': ask - bid,
                                'mid': (bid + ask) / 2
                            }
                            self.market_data[symbol].append(tick_data)
                            tick_counts[symbol] += 1
                            
                            # Display updates every 2 seconds
                            if current_time - last_display_time >= 2:
                                self.display_prices()
                                last_display_time = current_time
            
            time.sleep(0.1)
            
            # Show progress every 10 seconds
            elapsed = time.time() - start_time
            if elapsed > 0 and int(elapsed) % 10 == 0 and elapsed - int(elapsed) < 0.1:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: {int(elapsed)}/{duration_seconds}s")
                active_count = sum(1 for count in tick_counts.values() if count > 0)
                print(f"  Active symbols: {active_count}/{len(self.active_symbols)}")
                print(f"  Total ticks: {sum(tick_counts.values())}")
                print()
        
        print("="*70)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection complete.")
        
        # Show summary
        active_symbols = [sym for sym, count in tick_counts.items() if count > 0]
        if active_symbols:
            print(f"\nActive symbols ({len(active_symbols)}):")
            for symbol in active_symbols:
                print(f"  - {symbol}: {tick_counts[symbol]} ticks")
        
        # Check for WTI
        wti_found = False
        for symbol in active_symbols:
            if any(oil in symbol.upper() for oil in ['WTI', 'OIL', 'XTI', 'CL']):
                wti_found = True
                print(f"\n[SUCCESS] WTI Crude Oil found as: {symbol}")
                break
        
        if not wti_found:
            print("\n[INFO] WTI Crude Oil not available in your MT4")
            print("       Your broker may not offer oil trading or uses different symbols")
        
        return tick_counts
    
    def display_prices(self):
        """Display current prices"""
        timestamp_str = datetime.now().strftime('%H:%M:%S')
        print(f"\n[{timestamp_str}] Current Prices:")
        
        for symbol in self.active_symbols:
            if symbol in self.market_data and self.market_data[symbol]:
                latest = self.market_data[symbol][-1]
                bid = latest['bid']
                ask = latest['ask']
                spread = latest['spread']
                
                # Format based on price level
                if bid > 1000:
                    print(f"  {symbol:10} ${bid:.2f} / ${ask:.2f} (spread: ${spread:.2f})")
                elif bid > 100:
                    print(f"  {symbol:10} {bid:.3f} / {ask:.3f} (spread: {spread:.3f})")
                elif bid > 10:
                    print(f"  {symbol:10} {bid:.4f} / {ask:.4f} (spread: {spread:.4f})")
                else:
                    print(f"  {symbol:10} {bid:.6f} / {ask:.6f} (spread: {spread:.6f})")
    
    def analyze_data(self):
        """Analyze collected data"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analysis:")
        print("="*70)
        
        results = {}
        
        for symbol, data_list in self.market_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                
                print(f"\n{symbol}:")
                print(f"  Ticks: {len(df)}")
                
                # Price statistics
                bid_min = df['bid'].min()
                bid_max = df['bid'].max()
                bid_mean = df['bid'].mean()
                
                # Format based on price level
                if bid_mean > 1000:
                    print(f"  Bid: Min=${bid_min:.2f}, Max=${bid_max:.2f}, Mean=${bid_mean:.2f}")
                elif bid_mean > 100:
                    print(f"  Bid: Min={bid_min:.3f}, Max={bid_max:.3f}, Mean={bid_mean:.3f}")
                elif bid_mean > 10:
                    print(f"  Bid: Min={bid_min:.4f}, Max={bid_max:.4f}, Mean={bid_mean:.4f}")
                else:
                    print(f"  Bid: Min={bid_min:.6f}, Max={bid_max:.6f}, Mean={bid_mean:.6f}")
                
                # Volatility
                if len(df) > 1:
                    returns = df['mid'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * 100
                        print(f"  Volatility: {volatility:.4f}%")
                
                results[symbol] = df
        
        return results
    
    def save_data(self):
        """Save data to CSV files"""
        saved_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for symbol, data_list in self.market_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                filename = f"{symbol}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                saved_files.append(filename)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {len(df)} ticks to {filename}")
        
        # Save summary
        if self.active_symbols:
            with open(f'market_summary_{timestamp}.txt', 'w') as f:
                f.write(f"Market Data Collection Summary\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write("Available Symbols:\n")
                for symbol in self.active_symbols:
                    if self.market_data[symbol]:
                        f.write(f"  - {symbol}: {len(self.market_data[symbol])} ticks\n")
                
                # WTI status
                f.write("\nWTI Crude Oil Status:\n")
                wti_found = any(oil in sym.upper() for sym in self.active_symbols 
                              for oil in ['WTI', 'OIL', 'XTI'])
                if wti_found:
                    f.write("  [FOUND] WTI available for trading\n")
                else:
                    f.write("  [NOT FOUND] WTI not available in this MT4 account\n")
                    f.write("  Alternative commodities: GOLD available\n")
        
        return saved_files
    
    def disconnect(self):
        """Disconnect from MT4"""
        if self.dwx:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            
            # Unsubscribe from all symbols
            for symbol in self.active_symbols:
                try:
                    self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
                except:
                    pass
            
            # Shutdown connection
            self.dwx._DWX_ZMQ_SHUTDOWN_()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Disconnected from MT4")

def main():
    """Main function"""
    print("\n" + "="*70)
    print(" Market Data Collection via MT4 ")
    print("="*70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Searching for: WTI Crude Oil and all available markets")
    print("="*70)
    
    collector = MarketDataCollector()
    
    try:
        # Connect to MT4
        if collector.connect():
            # Discover available symbols
            found_symbols = collector.discover_symbols()
            
            if found_symbols:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Available markets:")
                for symbol, info in found_symbols.items():
                    print(f"  - {symbol}: {info['name']}")
                
                # Subscribe to all found symbols
                collector.subscribe_all()
                
                # Collect data
                print("\nPress Ctrl+C to stop early")
                tick_counts = collector.collect_data(duration_seconds=30)
                
                # Analyze data
                if any(tick_counts.values()):
                    results = collector.analyze_data()
                    
                    # Save data
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving data...")
                    saved_files = collector.save_data()
                else:
                    print("\n[WARNING] No data collected")
            else:
                print("\n[ERROR] No symbols available")
                print("Please check MT4 connection and market availability")
    
    except KeyboardInterrupt:
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure disconnection
        collector.disconnect()
    
    print("\n" + "="*70)
    print(" Collection Complete ")
    print("="*70)

if __name__ == "__main__":
    main()