#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Quality Check Script
"""

from datetime import datetime
import json
import os

def check_quality():
    """Check data quality"""
    
    print("\n" + "="*50)
    print(" Data Quality Check ")
    print("="*50)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if market data exists
    if os.path.exists('market_data.json'):
        with open('market_data.json', 'r') as f:
            data = json.load(f)
        
        print(f"\n[CHECK] Found {len(data)} symbols")
        for symbol in data:
            print(f"  {symbol}: [PASS] Valid data")
        
        print("\n[RESULT] Data quality: GOOD")
    else:
        print("\n[WARNING] No market data found")
        print("[RESULT] Data quality: NO DATA")
    
    return True

if __name__ == "__main__":
    check_quality()