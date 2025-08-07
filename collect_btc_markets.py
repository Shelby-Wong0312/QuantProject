#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Market Data Collection Script
"""

from datetime import datetime
import json
import time
import random

def collect_data():
    """Collect market data"""
    
    print("\n" + "="*50)
    print(" Market Data Collection ")
    print("="*50)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Simulate data collection
    symbols = ['BTCUSD', 'CRUDEOIL', 'EURUSD', 'GOLD']
    data = {}
    
    for symbol in symbols:
        # Generate fake data for testing
        data[symbol] = {
            'bid': round(random.uniform(1.0, 100.0), 5),
            'ask': round(random.uniform(1.0, 100.0), 5),
            'timestamp': datetime.now().isoformat()
        }
        print(f"  {symbol}: Collected")
    
    # Save data
    with open('market_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n[SUCCESS] Data saved to market_data.json")
    return True

if __name__ == "__main__":
    collect_data()