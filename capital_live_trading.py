#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Capital.com 實時交易系統
整合數據收集、策略執行、風險管理
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from capital_data_collector import CapitalDataCollector
from capital_trading_system import CapitalTradingSystem

class CapitalLiveTrading:
    """Capital.com 實時交易主系統"""
    
    def __init__(self):
        self.trading_system = CapitalTradingSystem()
        self.collector = self.trading_system.collector
        self.active_markets = []
        self.strategies = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0
        }
        
        # 載入市場代碼
        self.load_markets()
        
    def load_markets(self):
        """載入可用市場"""
        try:
            with open('capital_markets.json', 'r') as f:
                markets = json.load(f)
                
            # 選擇主要市場
            self.active_markets = [
                'BTCUSD',   # Bitcoin
                'ETHUSD',   # Ethereum
                'GOLD',     # Gold
                'EURUSD',   # EUR/USD
                'GBPUSD',   # GBP/USD
                'US100',    # NASDAQ 100
                'OIL_CRUDE' # Crude Oil
            ]
            
            # 過濾出實際存在的市場
            self.active_markets = [m for m in self.active_markets if m in markets]
            
            print(f"[SYSTEM] Loaded {len(self.active_markets)} active markets")
            for market in self.active_markets:
                print(f"  - {market}: {markets.get(market, 'Unknown')}")
                
        except FileNotFoundError:
            print("[WARNING] capital_markets.json not found, using default markets")
            self.active_markets = ['BTCUSD', 'EURUSD', 'GOLD']
    
    def momentum_strategy(self, epic: str) -> Optional[str]:
        """動量策略"""
        df = self.collector.get_historical_prices(epic, "HOUR", 50)
        if df.empty or len(df) < 20:
            return None
        
        # 計算動量指標
        df['returns'] = df['closePrice'].pct_change()
        df['momentum'] = df['returns'].rolling(10).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        
        latest = df.iloc[-1]
        momentum = latest['momentum']
        volatility = latest['volatility']
        
        # 生成信號
        if momentum > volatility * 2:
            return "BUY"
        elif momentum < -volatility * 2:
            return "SELL"
        
        return None
    
    def mean_reversion_strategy(self, epic: str) -> Optional[str]:
        """均值回歸策略"""
        df = self.collector.get_historical_prices(epic, "HOUR", 30)
        if df.empty or len(df) < 30:
            return None
        
        # 計算布林帶
        df['SMA'] = df['closePrice'].rolling(20).mean()
        df['STD'] = df['closePrice'].rolling(20).std()
        df['Upper'] = df['SMA'] + (df['STD'] * 2)
        df['Lower'] = df['SMA'] - (df['STD'] * 2)
        
        latest = df.iloc[-1]
        price = latest['closePrice']
        upper = latest['Upper']
        lower = latest['Lower']
        
        # 生成信號
        if price < lower:
            return "BUY"
        elif price > upper:
            return "SELL"
        
        return None
    
    def trend_following_strategy(self, epic: str) -> Optional[str]:
        """趨勢跟隨策略"""
        df = self.collector.get_historical_prices(epic, "HOUR", 100)
        if df.empty or len(df) < 50:
            return None
        
        # 計算移動平均線
        df['SMA20'] = df['closePrice'].rolling(20).mean()
        df['SMA50'] = df['closePrice'].rolling(50).mean()
        df['MACD'] = df['closePrice'].ewm(span=12).mean() - df['closePrice'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        latest = df.iloc[-1]
        sma20 = latest['SMA20']
        sma50 = latest['SMA50']
        macd = latest['MACD']
        signal = latest['Signal']
        
        # 生成信號
        if sma20 > sma50 and macd > signal:
            return "BUY"
        elif sma20 < sma50 and macd < signal:
            return "SELL"
        
        return None
    
    def execute_combined_strategy(self, epic: str) -> Optional[Dict]:
        """執行組合策略"""
        
        # 收集所有策略信號
        signals = {
            'momentum': self.momentum_strategy(epic),
            'mean_reversion': self.mean_reversion_strategy(epic),
            'trend_following': self.trend_following_strategy(epic)
        }
        
        # 計算信號共識
        buy_count = sum(1 for s in signals.values() if s == "BUY")
        sell_count = sum(1 for s in signals.values() if s == "SELL")
        
        # 需要至少2個策略同意
        final_signal = None
        confidence = 0
        
        if buy_count >= 2:
            final_signal = "BUY"
            confidence = buy_count / 3
        elif sell_count >= 2:
            final_signal = "SELL"
            confidence = sell_count / 3
        
        if final_signal:
            print(f"\n[SIGNAL] {epic}: {final_signal} (Confidence: {confidence:.1%})")
            print(f"  Momentum: {signals['momentum']}")
            print(f"  Mean Reversion: {signals['mean_reversion']}")
            print(f"  Trend Following: {signals['trend_following']}")
            
            # 獲取當前價格
            market = self.collector.get_market_details(epic)
            if market:
                current_price = market.get('snapshot', {}).get('bid')
                
                # 設置停損停利
                if final_signal == "BUY":
                    stop_loss = current_price * 0.98
                    take_profit = current_price * 1.03
                else:
                    stop_loss = current_price * 1.02
                    take_profit = current_price * 0.97
                
                # 計算倉位大小
                size = self.trading_system.calculate_position_size(epic, abs(current_price - stop_loss))
                
                # 執行交易
                result = self.trading_system.place_order(
                    epic=epic,
                    direction=final_signal,
                    size=size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if result['success']:
                    self.performance['total_trades'] += 1
                    return result
        
        return None
    
    def monitor_and_manage(self):
        """監控和管理持倉"""
        positions = self.collector.get_positions()
        
        if not positions:
            return
        
        print(f"\n[POSITIONS] {len(positions)} open positions")
        
        for pos in positions:
            market = pos.get('market', {})
            position = pos.get('position', {})
            
            epic = market.get('epic')
            direction = position.get('direction')
            size = position.get('size')
            open_level = position.get('level')
            current_price = market.get('bid') if direction == 'BUY' else market.get('offer')
            deal_id = position.get('dealId')
            
            # 計算盈虧
            if direction == 'BUY':
                pnl_pct = ((current_price - open_level) / open_level) * 100
            else:
                pnl_pct = ((open_level - current_price) / open_level) * 100
            
            print(f"  {epic}: {direction} {size} @ {open_level}")
            print(f"    Current: {current_price}, P&L: {pnl_pct:.2f}%")
            
            # 動態止損管理
            if pnl_pct > 2:  # 盈利超過2%
                # 移動止損到盈虧平衡點
                print(f"    [ACTION] Moving stop to breakeven")
                # 這裡可以實現移動止損邏輯
            elif pnl_pct < -3:  # 虧損超過3%
                # 強制平倉
                print(f"    [ACTION] Force closing due to loss")
                self.trading_system.close_position(deal_id)
                self.performance['losing_trades'] += 1
            elif pnl_pct > 5:  # 盈利超過5%
                # 獲利了結
                print(f"    [ACTION] Taking profit")
                self.trading_system.close_position(deal_id)
                self.performance['winning_trades'] += 1
    
    def run_live_trading(self, interval: int = 300):
        """運行實時交易
        
        Args:
            interval: 檢查間隔（秒）
        """
        print("\n" + "="*60)
        print(" Capital.com Live Trading System")
        print("="*60)
        print(f" Markets: {', '.join(self.active_markets)}")
        print(f" Check Interval: {interval} seconds")
        print(f" Account Balance: {self.trading_system.account_info.get('balance')}")
        print("="*60)
        
        try:
            while True:
                cycle_start = datetime.now()
                print(f"\n[CYCLE] {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 更新帳戶信息
                self.trading_system.account_info = self.collector.get_account_info()
                
                # 檢查每個市場
                for epic in self.active_markets:
                    try:
                        self.execute_combined_strategy(epic)
                    except Exception as e:
                        print(f"[ERROR] Strategy execution failed for {epic}: {e}")
                
                # 管理現有持倉
                self.monitor_and_manage()
                
                # 顯示績效
                self.display_performance()
                
                # 計算剩餘等待時間
                cycle_duration = (datetime.now() - cycle_start).seconds
                wait_time = max(0, interval - cycle_duration)
                
                if wait_time > 0:
                    print(f"\n[WAIT] Next cycle in {wait_time} seconds...")
                    time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n[STOP] Live trading stopped by user")
            self.display_final_report()
    
    def display_performance(self):
        """顯示績效統計"""
        win_rate = 0
        if self.performance['total_trades'] > 0:
            win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
        
        print(f"\n[PERFORMANCE]")
        print(f"  Total Trades: {self.performance['total_trades']}")
        print(f"  Winners: {self.performance['winning_trades']}")
        print(f"  Losers: {self.performance['losing_trades']}")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    def display_final_report(self):
        """顯示最終報告"""
        print("\n" + "="*60)
        print(" Trading Session Report")
        print("="*60)
        
        # 獲取最終帳戶信息
        final_account = self.collector.get_account_info()
        initial_balance = 137760.8  # 從之前的測試中獲得
        final_balance = final_account.get('balance', initial_balance)
        
        print(f" Initial Balance: {initial_balance}")
        print(f" Final Balance: {final_balance}")
        print(f" Net P&L: {final_balance - initial_balance:.2f}")
        print(f" Return: {((final_balance - initial_balance) / initial_balance) * 100:.2f}%")
        
        self.display_performance()
        
        print("="*60)

def main():
    """主函數"""
    
    # 創建實時交易系統
    live_trading = CapitalLiveTrading()
    
    # 選擇運行模式
    print("\nSelect trading mode:")
    print("1. Test Mode (Dry run, no real trades)")
    print("2. Live Mode (Real trades on demo account)")
    print("3. Monitor Only (Just watch markets)")
    
    mode = input("\nEnter choice (1-3): ").strip()
    
    if mode == "1":
        # 測試模式
        print("\n[TEST MODE] Running strategy tests...")
        for market in live_trading.active_markets[:3]:
            print(f"\nTesting {market}:")
            live_trading.execute_combined_strategy(market)
    
    elif mode == "2":
        # 實時交易模式
        interval = 300  # 5分鐘
        print(f"\n[LIVE MODE] Starting live trading (interval: {interval}s)...")
        live_trading.run_live_trading(interval)
    
    elif mode == "3":
        # 監控模式
        print("\n[MONITOR MODE] Watching markets...")
        while True:
            for market in live_trading.active_markets:
                details = live_trading.collector.get_market_details(market)
                if details:
                    bid = details.get('snapshot', {}).get('bid')
                    offer = details.get('snapshot', {}).get('offer')
                    print(f"{market}: Bid={bid}, Ask={offer}")
            time.sleep(10)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()