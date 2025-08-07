#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檢查比特幣持倉狀態
"""

import os
import sys
from datetime import datetime
from capital_data_collector import CapitalDataCollector

def check_bitcoin_position():
    """檢查比特幣持倉"""
    
    print("\n" + "="*60)
    print(" Bitcoin Position Status")
    print("="*60)
    print(f" Time: {datetime.now()}")
    print("="*60)
    
    # 創建數據收集器
    collector = CapitalDataCollector()
    
    # 登入
    if not collector.login():
        print("[ERROR] Failed to login")
        return
    
    # 獲取帳戶信息
    print("\n[1] Account Status:")
    account = collector.get_account_info()
    if account:
        print(f"  Balance: ${account.get('balance'):,.2f}")
        print(f"  Available: ${account.get('available'):,.2f}")
        print(f"  P&L: ${account.get('profitLoss'):,.2f}")
    
    # 獲取所有持倉
    print("\n[2] Current Positions:")
    positions = collector.get_positions()
    
    if not positions:
        print("  No open positions")
        return
    
    # 查找比特幣持倉
    btc_found = False
    for pos in positions:
        market = pos.get('market', {})
        position = pos.get('position', {})
        
        epic = market.get('epic')
        
        if epic == 'BTCUSD':
            btc_found = True
            
            # 提取持倉信息
            instrument = market.get('instrumentName')
            direction = position.get('direction')
            size = position.get('size')
            open_level = position.get('level')
            deal_id = position.get('dealId')
            created = position.get('createdDateUTC')
            
            # 當前價格
            bid = market.get('bid')
            offer = market.get('offer')
            current_price = bid if direction == 'BUY' else offer
            
            # 計算盈虧
            if direction == 'BUY':
                pnl = (current_price - open_level) * size
                pnl_pct = ((current_price - open_level) / open_level) * 100
            else:
                pnl = (open_level - current_price) * size
                pnl_pct = ((open_level - current_price) / open_level) * 100
            
            print("\n  [BITCOIN POSITION FOUND]")
            print("  " + "-"*40)
            print(f"  Instrument: {instrument}")
            print(f"  Deal ID: {deal_id}")
            print(f"  Direction: {direction}")
            print(f"  Size: {size} BTC")
            print(f"  Open Price: ${open_level:,.2f}")
            print(f"  Current Bid: ${bid:,.2f}")
            print(f"  Current Ask: ${offer:,.2f}")
            print(f"  Created: {created}")
            print("\n  [P&L Calculation]")
            print(f"  Current Value: ${current_price * size:,.2f}")
            print(f"  P&L: ${pnl:,.2f}")
            print(f"  P&L %: {pnl_pct:+.2f}%")
            
            if pnl > 0:
                print(f"  Status: PROFIT")
            elif pnl < 0:
                print(f"  Status: LOSS")
            else:
                print(f"  Status: BREAKEVEN")
            
            # 停損停利信息
            stop_level = position.get('stopLevel')
            limit_level = position.get('limitLevel')
            
            if stop_level or limit_level:
                print("\n  [Risk Management]")
                if stop_level:
                    print(f"  Stop Loss: ${stop_level:,.2f}")
                if limit_level:
                    print(f"  Take Profit: ${limit_level:,.2f}")
    
    if not btc_found:
        print("  No Bitcoin position found")
    
    # 獲取最新比特幣價格
    print("\n[3] Current Bitcoin Market:")
    market = collector.get_market_details("BTCUSD")
    if market:
        snapshot = market.get('snapshot', {})
        print(f"  Bid: ${snapshot.get('bid'):,.2f}")
        print(f"  Ask: ${snapshot.get('offer'):,.2f}")
        print(f"  High: ${snapshot.get('high'):,.2f}")
        print(f"  Low: ${snapshot.get('low'):,.2f}")
        print(f"  Net Change: ${snapshot.get('netChange'):,.2f}")
        print(f"  % Change: {snapshot.get('percentageChange'):.2f}%")
        print(f"  Market Status: {snapshot.get('marketStatus')}")
    
    print("\n" + "="*60)
    print(" Position Check Complete")
    print("="*60)

if __name__ == "__main__":
    check_bitcoin_position()