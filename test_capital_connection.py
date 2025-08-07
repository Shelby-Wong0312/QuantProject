#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試Capital.com連接並找出正確的市場代碼
"""

import os
import sys
import requests
from dotenv import load_dotenv
from capital_data_collector import CapitalDataCollector

def test_capital_connection():
    """測試Capital.com連接"""
    
    print("\n" + "="*60)
    print(" Capital.com Connection Test")
    print("="*60)
    
    # 創建收集器
    collector = CapitalDataCollector()
    
    # 測試登入
    print("\n[1] Testing Login...")
    if collector.login():
        print("  [PASS] Successfully logged in")
    else:
        print("  [FAIL] Login failed")
        return False
    
    # 獲取帳戶信息
    print("\n[2] Getting Account Info...")
    account = collector.get_account_info()
    if account:
        print(f"  [PASS] Account ID: {account.get('accountId')}")
        print(f"  Balance: {account.get('balance')} {account.get('currency')}")
    else:
        print("  [FAIL] Could not get account info")
    
    # 搜索常用市場
    print("\n[3] Finding Popular Markets...")
    
    # 測試不同的搜索詞
    search_terms = {
        "Crypto": ["Bitcoin", "Ethereum", "BTC"],
        "Forex": ["EUR", "USD", "GBP"],
        "Commodities": ["Gold", "Oil", "Silver"],
        "Indices": ["US", "SP", "DAX"]
    }
    
    available_epics = {}
    
    for category, terms in search_terms.items():
        print(f"\n  {category}:")
        for term in terms:
            markets = collector.get_markets(term)
            if markets:
                for market in markets[:3]:  # 顯示前3個
                    epic = market.get('epic')
                    name = market.get('instrumentName')
                    if epic and epic not in available_epics:
                        available_epics[epic] = name
                        print(f"    - {name}: {epic}")
    
    # 測試價格獲取
    print("\n[4] Testing Price Data...")
    
    # 選擇一些找到的市場進行測試
    test_epics = list(available_epics.keys())[:5]
    
    success_count = 0
    for epic in test_epics:
        name = available_epics[epic]
        print(f"\n  Testing {name} ({epic}):")
        
        # 獲取市場詳情
        details = collector.get_market_details(epic)
        if details:
            market_status = details.get('snapshot', {}).get('marketStatus')
            bid = details.get('snapshot', {}).get('bid')
            offer = details.get('snapshot', {}).get('offer')
            
            print(f"    Market Status: {market_status}")
            print(f"    Bid: {bid}")
            print(f"    Offer: {offer}")
            
            if bid and offer:
                success_count += 1
                print(f"    [PASS] Price data available")
            else:
                print(f"    [WARNING] No price data")
        else:
            print(f"    [FAIL] Could not get market details")
    
    # 測試歷史數據
    print("\n[5] Testing Historical Data...")
    
    if test_epics:
        epic = test_epics[0]
        print(f"  Getting historical data for {available_epics[epic]} ({epic})...")
        
        # 嘗試不同的resolution
        resolutions = ["MINUTE", "MINUTE_5", "HOUR"]
        
        for resolution in resolutions:
            df = collector.get_historical_prices(epic, resolution, 10)
            if not df.empty:
                print(f"    [PASS] {resolution} data available ({len(df)} points)")
                print(f"    Latest: {df.index[-1]} - Close: {df['closePrice'].iloc[-1]}")
                break
            else:
                print(f"    [FAIL] No {resolution} data")
    
    # 總結
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)
    print(f"  Login: {'PASS' if collector.cst else 'FAIL'}")
    print(f"  Markets Found: {len(available_epics)}")
    print(f"  Price Data Success: {success_count}/{len(test_epics)}")
    print("="*60)
    
    # 保存找到的市場代碼
    if available_epics:
        import json
        with open('capital_markets.json', 'w') as f:
            json.dump(available_epics, f, indent=2)
        print(f"\n[INFO] Saved {len(available_epics)} market codes to capital_markets.json")
    
    return True

if __name__ == "__main__":
    test_capital_connection()