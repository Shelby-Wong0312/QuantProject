#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Capital.com 完整交易系統
整合數據收集、交易執行、風險管理
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from capital_data_collector import CapitalDataCollector

# Load environment variables
load_dotenv()

class CapitalTradingSystem:
    """Capital.com 交易系統"""
    
    def __init__(self):
        self.collector = CapitalDataCollector()
        self.positions = {}
        self.orders = []
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 每筆交易風險2%
        self.account_info = {}
        
        # 登入
        if self.collector.login():
            self.account_info = self.collector.get_account_info()
            print(f"[SYSTEM] Trading system initialized")
            print(f"[ACCOUNT] Balance: {self.account_info.get('balance')} {self.account_info.get('currency')}")
        else:
            print("[ERROR] Failed to initialize trading system")
    
    def place_order(self, epic: str, direction: str, size: float, 
                   stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None) -> Dict:
        """下單
        
        Args:
            epic: 市場代碼
            direction: BUY 或 SELL
            size: 交易量
            stop_loss: 停損價格
            take_profit: 停利價格
        """
        if not self.collector.cst:
            return {"success": False, "error": "Not logged in"}
        
        # 檢查持倉數量
        current_positions = self.collector.get_positions()
        if len(current_positions) >= self.max_positions:
            return {"success": False, "error": f"Maximum positions ({self.max_positions}) reached"}
        
        # 建立訂單請求
        positions_url = f"{self.collector.base_url}/positions"
        headers = {
            "CST": self.collector.cst,
            "X-SECURITY-TOKEN": self.collector.x_security_token,
            "Content-Type": "application/json"
        }
        
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": False,
            "trailingStop": False
        }
        
        # 添加停損停利
        if stop_loss:
            payload["stopLevel"] = stop_loss
        if take_profit:
            payload["profitLevel"] = take_profit
        
        try:
            response = self.collector.session.post(
                positions_url, 
                headers=headers, 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                deal_reference = result.get('dealReference')
                
                # 記錄訂單
                order = {
                    'dealReference': deal_reference,
                    'epic': epic,
                    'direction': direction,
                    'size': size,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPENED'
                }
                self.orders.append(order)
                
                print(f"[ORDER] {direction} {size} {epic} - Deal: {deal_reference}")
                return {"success": True, "data": result}
            else:
                error_data = response.json()
                print(f"[ERROR] Order failed: {error_data}")
                return {"success": False, "error": error_data}
                
        except Exception as e:
            print(f"[ERROR] Order exception: {e}")
            return {"success": False, "error": str(e)}
    
    def close_position(self, deal_id: str) -> Dict:
        """平倉"""
        if not self.collector.cst:
            return {"success": False, "error": "Not logged in"}
        
        close_url = f"{self.collector.base_url}/positions/{deal_id}"
        headers = {
            "CST": self.collector.cst,
            "X-SECURITY-TOKEN": self.collector.x_security_token,
            "_method": "DELETE"
        }
        
        try:
            response = self.collector.session.post(
                close_url, 
                headers=headers, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[CLOSE] Position {deal_id} closed")
                return {"success": True, "data": result}
            else:
                error_data = response.json()
                return {"success": False, "error": error_data}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_position_size(self, epic: str, stop_loss_pips: float) -> float:
        """計算倉位大小（基於風險管理）"""
        if not self.account_info:
            return 0.01  # 默認最小倉位
        
        balance = self.account_info.get('balance', 10000)
        risk_amount = balance * self.risk_per_trade
        
        # 獲取市場詳情
        market = self.collector.get_market_details(epic)
        if not market:
            return 0.01
        
        # 計算每點價值
        pip_value = market.get('snapshot', {}).get('onePipMeans', 1)
        
        # 計算倉位大小
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # 確保符合最小/最大限制
        min_size = market.get('dealingRules', {}).get('minDealSize', {}).get('value', 0.01)
        max_size = market.get('dealingRules', {}).get('maxDealSize', {}).get('value', 100)
        
        position_size = max(min_size, min(position_size, max_size))
        
        return round(position_size, 2)
    
    def execute_strategy(self, epic: str) -> Optional[Dict]:
        """執行交易策略（簡單動量策略）"""
        
        # 獲取歷史數據
        df = self.collector.get_historical_prices(epic, "HOUR", 50)
        if df.empty:
            print(f"[STRATEGY] No data for {epic}")
            return None
        
        # 計算指標
        df['SMA20'] = df['closePrice'].rolling(20).mean()
        df['SMA50'] = df['closePrice'].rolling(50).mean()
        df['RSI'] = self.calculate_rsi(df['closePrice'])
        
        # 獲取最新值
        latest = df.iloc[-1]
        current_price = latest['closePrice']
        sma20 = latest['SMA20']
        sma50 = latest['SMA50']
        rsi = latest['RSI']
        
        # 生成信號
        signal = None
        if sma20 > sma50 and rsi < 70 and current_price > sma20:
            signal = "BUY"
            stop_loss = current_price * 0.98  # 2% 停損
            take_profit = current_price * 1.04  # 4% 停利
        elif sma20 < sma50 and rsi > 30 and current_price < sma20:
            signal = "SELL"
            stop_loss = current_price * 1.02  # 2% 停損
            take_profit = current_price * 0.96  # 4% 停利
        
        if signal:
            # 計算倉位大小
            stop_loss_pips = abs(current_price - stop_loss) * 100
            size = self.calculate_position_size(epic, stop_loss_pips)
            
            # 下單
            result = self.place_order(
                epic=epic,
                direction=signal,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result['success']:
                print(f"[STRATEGY] {signal} signal executed for {epic}")
                return result
            else:
                print(f"[STRATEGY] Failed to execute {signal} for {epic}")
                return None
        
        return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def monitor_positions(self):
        """監控持倉"""
        positions = self.collector.get_positions()
        
        if not positions:
            print("[MONITOR] No open positions")
            return
        
        print(f"\n[MONITOR] Open Positions ({len(positions)}):")
        print("-" * 60)
        
        total_pnl = 0
        for pos in positions:
            market = pos.get('market', {})
            position = pos.get('position', {})
            
            epic = market.get('epic')
            name = market.get('instrumentName')
            direction = position.get('direction')
            size = position.get('size')
            open_level = position.get('level')
            current_price = market.get('bid') if direction == 'BUY' else market.get('offer')
            
            # 計算盈虧
            if direction == 'BUY':
                pnl = (current_price - open_level) * size
            else:
                pnl = (open_level - current_price) * size
            
            total_pnl += pnl
            
            print(f"  {name} ({epic})")
            print(f"    Direction: {direction}, Size: {size}")
            print(f"    Open: {open_level}, Current: {current_price}")
            print(f"    P&L: {pnl:.2f} {'✓' if pnl > 0 else '✗'}")
            print()
        
        print(f"  Total P&L: {total_pnl:.2f}")
        print("-" * 60)
    
    def run_automated_trading(self, epics: List[str], interval: int = 300):
        """運行自動交易
        
        Args:
            epics: 要交易的市場列表
            interval: 檢查間隔（秒）
        """
        print(f"\n[AUTO] Starting automated trading for {epics}")
        print(f"[AUTO] Check interval: {interval} seconds")
        
        try:
            while True:
                # 更新帳戶信息
                self.account_info = self.collector.get_account_info()
                
                # 檢查每個市場
                for epic in epics:
                    print(f"\n[CHECK] Analyzing {epic}...")
                    self.execute_strategy(epic)
                
                # 監控持倉
                self.monitor_positions()
                
                # 等待下次檢查
                print(f"\n[WAIT] Next check in {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n[STOP] Automated trading stopped")

def test_capital_trading():
    """測試Capital.com交易系統"""
    print("\n" + "="*60)
    print(" Capital.com Trading System Test")
    print("="*60)
    
    # 創建交易系統
    system = CapitalTradingSystem()
    
    # 測試下單（小量測試）
    print("\n[TEST] Placing test order...")
    result = system.place_order(
        epic="BITCOIN",
        direction="BUY",
        size=0.01  # 最小單位
    )
    
    if result['success']:
        print("[SUCCESS] Test order placed")
        
        # 等待3秒
        time.sleep(3)
        
        # 獲取持倉
        positions = system.collector.get_positions()
        if positions:
            # 平掉第一個持倉
            deal_id = positions[0].get('position', {}).get('dealId')
            if deal_id:
                print(f"\n[TEST] Closing position {deal_id}...")
                close_result = system.close_position(deal_id)
                if close_result['success']:
                    print("[SUCCESS] Position closed")
    
    # 監控持倉
    print("\n[TEST] Current positions:")
    system.monitor_positions()
    
    print("\n" + "="*60)
    print(" Test Complete")
    print("="*60)

if __name__ == "__main__":
    # 運行測試
    test_capital_trading()
    
    # 如果要運行自動交易，取消下面的註釋
    # system = CapitalTradingSystem()
    # system.run_automated_trading(["BITCOIN", "GOLD", "EURUSD"], interval=300)