#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect Available Cryptocurrency Data from MT4
Based on discovered symbols: GOLD, Ethereum, LTCUSD, XRPUSD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
from datetime import datetime
import time
import pandas as pd
import json

class CryptoCollector:
    def __init__(self):
        self.dwx = None
        self.crypto_data = {
            'XRPUSD': [],
            'Ethereum': [],
            'LTCUSD': [],
            'GOLD': []
        }
        self.last_prices = {}
        
    def connect(self):
        """Connect to MT4"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to MT4...")
        
        self.dwx = DWX_ZeroMQ_Connector(
            _ClientID='CryptoCollector',
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
    
    def subscribe_all(self):
        """Subscribe to all available crypto/commodity symbols"""
        symbols = ['XRPUSD', 'Ethereum', 'LTCUSD', 'GOLD']
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Subscribing to available symbols...")
        
        for symbol in symbols:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Subscribing to {symbol}...")
            self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
            time.sleep(0.5)
    
    def collect_data(self, duration_seconds=60):
        """Collect data for specified duration"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting data collection for {duration_seconds} seconds...")
        print("="*70)
        
        start_time = time.time()
        last_display_time = start_time
        tick_counts = {symbol: 0 for symbol in self.crypto_data.keys()}
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Check market data
            if self.dwx._Market_Data_DB:
                for symbol, data in self.dwx._Market_Data_DB.items():
                    if data and symbol in self.crypto_data:
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
                            self.crypto_data[symbol].append(tick_data)
                            tick_counts[symbol] += 1
                            
                            # Calculate price change
                            if symbol in self.last_prices:
                                change = bid - self.last_prices[symbol]
                                change_str = f"[{change:+.6f}]" if abs(change) > 0 else "[=]"
                            else:
                                change_str = "[NEW]"
                            
                            self.last_prices[symbol] = bid
                            
                            # Display updates every second
                            if current_time - last_display_time >= 1:
                                timestamp_str = datetime.now().strftime('%H:%M:%S')
                                
                                if symbol == 'GOLD':
                                    print(f"[{timestamp_str}] GOLD:     ${bid:.2f} / ${ask:.2f} | Spread: ${ask-bid:.2f} {change_str}")
                                elif symbol == 'Ethereum':
                                    print(f"[{timestamp_str}] Ethereum: ${bid:.2f} / ${ask:.2f} | Spread: ${ask-bid:.2f} {change_str}")
                                elif symbol == 'LTCUSD':
                                    print(f"[{timestamp_str}] Litecoin: ${bid:.2f} / ${ask:.2f} | Spread: ${ask-bid:.2f} {change_str}")
                                elif symbol == 'XRPUSD':
                                    print(f"[{timestamp_str}] Ripple:   ${bid:.6f} / ${ask:.6f} | Spread: ${ask-bid:.6f} {change_str}")
                                
                                last_display_time = current_time
            
            time.sleep(0.1)
            
            # Show progress every 10 seconds
            elapsed = time.time() - start_time
            if elapsed > 0 and int(elapsed) % 10 == 0 and elapsed - int(elapsed) < 0.1:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: {int(elapsed)}/{duration_seconds}s")
                print(f"  Ticks collected - XRP: {tick_counts['XRPUSD']}, ETH: {tick_counts['Ethereum']}, LTC: {tick_counts['LTCUSD']}, GOLD: {tick_counts['GOLD']}")
                print()
        
        print("="*70)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection complete.")
        print(f"  Total ticks: XRP={tick_counts['XRPUSD']}, ETH={tick_counts['Ethereum']}, LTC={tick_counts['LTCUSD']}, GOLD={tick_counts['GOLD']}")
        
        return tick_counts
    
    def analyze_data(self):
        """Analyze collected data"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analysis:")
        print("="*70)
        
        results = {}
        
        for symbol, data_list in self.crypto_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                
                print(f"\n{symbol}:")
                print(f"  Ticks: {len(df)}")
                print(f"  Time range: {df['timestamp'].min().strftime('%H:%M:%S')} to {df['timestamp'].max().strftime('%H:%M:%S')}")
                
                if symbol == 'XRPUSD':
                    print(f"  Bid  - Min: ${df['bid'].min():.6f} | Max: ${df['bid'].max():.6f} | Mean: ${df['bid'].mean():.6f}")
                    print(f"  Ask  - Min: ${df['ask'].min():.6f} | Max: ${df['ask'].max():.6f} | Mean: ${df['ask'].mean():.6f}")
                    print(f"  Spread - Min: ${df['spread'].min():.6f} | Max: ${df['spread'].max():.6f} | Mean: ${df['spread'].mean():.6f}")
                else:
                    print(f"  Bid  - Min: ${df['bid'].min():.2f} | Max: ${df['bid'].max():.2f} | Mean: ${df['bid'].mean():.2f}")
                    print(f"  Ask  - Min: ${df['ask'].min():.2f} | Max: ${df['ask'].max():.2f} | Mean: ${df['ask'].mean():.2f}")
                    print(f"  Spread - Min: ${df['spread'].min():.2f} | Max: ${df['spread'].max():.2f} | Mean: ${df['spread'].mean():.2f}")
                
                # Calculate volatility
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
        
        for symbol, data_list in self.crypto_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                filename = f"{symbol.lower().replace('/', '_')}_data.csv"
                df.to_csv(filename, index=False)
                saved_files.append(filename)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {len(df)} ticks to {filename}")
        
        return saved_files
    
    def disconnect(self):
        """Disconnect from MT4"""
        if self.dwx:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            
            # Unsubscribe from all symbols
            for symbol in self.crypto_data.keys():
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
    print(" Cryptocurrency & Commodity Data Collection via MT4 ")
    print("="*70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Available symbols: XRPUSD (Ripple), Ethereum, LTCUSD (Litecoin), GOLD")
    print("="*70)
    
    collector = CryptoCollector()
    
    try:
        # Connect to MT4
        if collector.connect():
            # Subscribe to available symbols
            collector.subscribe_all()
            
            # Collect data for 30 seconds
            print("\nPress Ctrl+C to stop early")
            tick_counts = collector.collect_data(duration_seconds=30)
            
            # Analyze data
            results = collector.analyze_data()
            
            # Save data
            if any(tick_counts.values()):
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving data...")
                saved_files = collector.save_data()
                
                # Show sample of last ticks
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Last 3 ticks per symbol:")
                for symbol, df in results.items():
                    if df is not None and len(df) > 0:
                        print(f"\n{symbol}:")
                        print(df[['timestamp', 'bid', 'ask', 'spread']].tail(3).to_string(index=False))
            else:
                print("\n[WARNING] No data was collected")
                print("Please check MT4 connection and symbol availability")
    
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
    print("\nNote: Bitcoin (BTCUSD) is not available in your MT4.")
    print("Your broker may not offer Bitcoin trading or uses a different symbol.")
    print("\nAlternatives collected:")
    print("- Ripple (XRP) - Major cryptocurrency")
    print("- Ethereum (ETH) - Second largest cryptocurrency")
    print("- Litecoin (LTC) - Popular altcoin")
    print("- Gold - Traditional safe haven asset")

if __name__ == "__main__":
    main()