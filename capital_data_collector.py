#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Capital.com 實時數據收集系統
使用REST API獲取市場數據
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

class CapitalDataCollector:
    """Capital.com 數據收集器"""
    
    def __init__(self):
        self.api_key = os.getenv('CAPITAL_API_KEY', '').strip('"')
        self.identifier = os.getenv('CAPITAL_IDENTIFIER', '').strip('"')
        self.password = os.getenv('CAPITAL_API_PASSWORD', '').strip('"')
        self.demo_mode = os.getenv('CAPITAL_DEMO_MODE', 'True').lower() == 'true'
        
        if self.demo_mode:
            self.base_url = "https://demo-api-capital.backend-capital.com/api/v1"
        else:
            self.base_url = "https://api-capital.backend-capital.com/api/v1"
            
        self.cst = None
        self.x_security_token = None
        self.session = requests.Session()
        self.market_data = {}
        self.running = False
        
        print(f"\n[INFO] Capital.com Data Collector initialized")
        print(f"[INFO] Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
    
    def login(self) -> bool:
        """登入 Capital.com API"""
        login_url = f"{self.base_url}/session"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "identifier": self.identifier,
            "password": self.password
        }
        
        try:
            response = self.session.post(login_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                print("[SUCCESS] Logged in to Capital.com")
                return True
            else:
                print(f"[ERROR] Login failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Login exception: {e}")
            return False
    
    def get_markets(self, search_term: str = "") -> List[Dict]:
        """獲取可用市場列表"""
        if not self.cst and not self.login():
            return []
        
        markets_url = f"{self.base_url}/markets"
        if search_term:
            markets_url += f"?searchTerm={search_term}"
        
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token
        }
        
        try:
            response = self.session.get(markets_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('markets', [])
            else:
                print(f"[ERROR] Failed to get markets: {response.status_code}")
                return []
        except Exception as e:
            print(f"[ERROR] Get markets exception: {e}")
            return []
    
    def get_market_details(self, epic: str) -> Dict:
        """獲取市場詳細信息"""
        if not self.cst and not self.login():
            return {}
        
        market_url = f"{self.base_url}/markets/{epic}"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token
        }
        
        try:
            response = self.session.get(market_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] Failed to get market details: {response.status_code}")
                return {}
        except Exception as e:
            print(f"[ERROR] Get market details exception: {e}")
            return {}
    
    def get_price(self, epic: str) -> Dict:
        """獲取即時價格"""
        if not self.cst and not self.login():
            return {}
        
        price_url = f"{self.base_url}/prices/{epic}"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token
        }
        
        try:
            response = self.session.get(price_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                price_data = {
                    'epic': epic,
                    'bid': data.get('bid'),
                    'offer': data.get('offer'),
                    'timestamp': datetime.now().isoformat()
                }
                self.market_data[epic] = price_data
                return price_data
            else:
                return {}
        except Exception as e:
            print(f"[ERROR] Get price exception: {e}")
            return {}
    
    def get_historical_prices(self, epic: str, resolution: str = "HOUR", max_points: int = 100) -> pd.DataFrame:
        """獲取歷史價格數據
        
        Args:
            epic: 市場代碼
            resolution: MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK
            max_points: 最多返回的數據點數
        """
        if not self.cst and not self.login():
            return pd.DataFrame()
        
        # 計算時間範圍
        now = datetime.now()
        if resolution == "MINUTE":
            from_date = now - timedelta(minutes=max_points)
        elif resolution == "HOUR":
            from_date = now - timedelta(hours=max_points)
        elif resolution == "DAY":
            from_date = now - timedelta(days=max_points)
        else:
            from_date = now - timedelta(hours=max_points)
        
        prices_url = f"{self.base_url}/prices/{epic}"
        params = {
            "resolution": resolution,
            "max": max_points,
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": now.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token
        }
        
        try:
            response = self.session.get(prices_url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                
                if prices:
                    df = pd.DataFrame(prices)
                    df['timestamp'] = pd.to_datetime(df['snapshotTime'])
                    df = df.set_index('timestamp')
                    return df[['openPrice', 'highPrice', 'lowPrice', 'closePrice']]
                else:
                    return pd.DataFrame()
            else:
                print(f"[ERROR] Failed to get historical prices: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] Get historical prices exception: {e}")
            return pd.DataFrame()
    
    def start_real_time_collection(self, epics: List[str], interval: int = 1):
        """開始實時數據收集
        
        Args:
            epics: 要監控的市場代碼列表
            interval: 更新間隔(秒)
        """
        self.running = True
        
        def collect_loop():
            while self.running:
                for epic in epics:
                    price = self.get_price(epic)
                    if price:
                        print(f"[DATA] {epic}: Bid={price.get('bid')}, Ask={price.get('offer')}")
                
                time.sleep(interval)
        
        # 在後台線程中運行
        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()
        print(f"[INFO] Started real-time collection for {len(epics)} markets")
    
    def stop_collection(self):
        """停止數據收集"""
        self.running = False
        print("[INFO] Stopped data collection")
    
    def get_account_info(self) -> Dict:
        """獲取帳戶信息"""
        if not self.cst and not self.login():
            return {}
        
        accounts_url = f"{self.base_url}/accounts"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token
        }
        
        try:
            response = self.session.get(accounts_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                accounts = data.get('accounts', [])
                if accounts:
                    account = accounts[0]  # 使用第一個帳戶
                    return {
                        'accountId': account.get('accountId'),
                        'accountName': account.get('accountName'),
                        'balance': account.get('balance', {}).get('balance'),
                        'available': account.get('balance', {}).get('available'),
                        'profitLoss': account.get('balance', {}).get('profitLoss'),
                        'currency': account.get('currency')
                    }
                return {}
            else:
                print(f"[ERROR] Failed to get account info: {response.status_code}")
                return {}
        except Exception as e:
            print(f"[ERROR] Get account info exception: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """獲取當前持倉"""
        if not self.cst and not self.login():
            return []
        
        positions_url = f"{self.base_url}/positions"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token
        }
        
        try:
            response = self.session.get(positions_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('positions', [])
            else:
                return []
        except Exception as e:
            print(f"[ERROR] Get positions exception: {e}")
            return []

def main():
    """測試 Capital.com 數據收集"""
    print("\n" + "="*60)
    print(" Capital.com Data Collection Test")
    print("="*60)
    
    # 創建收集器
    collector = CapitalDataCollector()
    
    # 登入
    if not collector.login():
        print("[ERROR] Failed to login")
        return
    
    # 獲取帳戶信息
    print("\n[1] Account Information:")
    account = collector.get_account_info()
    if account:
        print(f"  Account ID: {account.get('accountId')}")
        print(f"  Balance: {account.get('balance')} {account.get('currency')}")
        print(f"  Available: {account.get('available')}")
        print(f"  P&L: {account.get('profitLoss')}")
    
    # 搜索市場
    print("\n[2] Available Markets (Bitcoin):")
    markets = collector.get_markets("Bitcoin")
    for market in markets[:5]:  # 顯示前5個
        print(f"  - {market.get('instrumentName')} ({market.get('epic')})")
    
    # 測試價格獲取
    test_epics = ["BITCOIN", "GOLD", "EURUSD"]
    print(f"\n[3] Real-time Prices:")
    
    for epic in test_epics:
        price = collector.get_price(epic)
        if price:
            print(f"  {epic}: Bid={price.get('bid')}, Ask={price.get('offer')}")
    
    # 獲取歷史數據
    print("\n[4] Historical Data (BITCOIN - Last 10 hours):")
    df = collector.get_historical_prices("BITCOIN", "HOUR", 10)
    if not df.empty:
        print(df.tail())
    
    # 開始實時收集
    print("\n[5] Starting real-time collection (10 seconds)...")
    collector.start_real_time_collection(["BITCOIN", "GOLD"], interval=2)
    
    # 運行10秒
    time.sleep(10)
    
    # 停止收集
    collector.stop_collection()
    
    print("\n[6] Current positions:")
    positions = collector.get_positions()
    if positions:
        for pos in positions:
            print(f"  - {pos.get('market', {}).get('instrumentName')}: {pos.get('position', {}).get('size')}")
    else:
        print("  No open positions")
    
    print("\n" + "="*60)
    print(" Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()