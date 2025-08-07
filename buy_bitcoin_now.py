#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
立即購買比特幣
"""

import os
import sys
from datetime import datetime
from capital_trading_system import CapitalTradingSystem

def buy_bitcoin():
    """執行比特幣購買訂單"""
    
    print("\n" + "="*60)
    print(" Executing Bitcoin Purchase Order")
    print("="*60)
    print(f" Time: {datetime.now()}")
    print("="*60)
    
    # 創建交易系統
    print("\n[1] Initializing trading system...")
    system = CapitalTradingSystem()
    
    # 獲取當前比特幣價格
    print("\n[2] Getting current Bitcoin price...")
    market = system.collector.get_market_details("BTCUSD")
    
    if market:
        bid = market.get('snapshot', {}).get('bid')
        offer = market.get('snapshot', {}).get('offer')
        spread = offer - bid
        
        print(f"  Market: Bitcoin/USD (BTCUSD)")
        print(f"  Bid Price: ${bid:,.2f}")
        print(f"  Ask Price: ${offer:,.2f}")
        print(f"  Spread: ${spread:.2f}")
        
        # 計算購買1個比特幣需要的資金
        total_cost = offer * 1  # 1 BTC
        print(f"\n[3] Order Details:")
        print(f"  Amount: 1.0 BTC")
        print(f"  Price: ${offer:,.2f}")
        print(f"  Total Cost: ${total_cost:,.2f}")
        
        # 檢查帳戶餘額
        account = system.collector.get_account_info()
        balance = account.get('balance', 0)
        print(f"\n[4] Account Check:")
        print(f"  Current Balance: ${balance:,.2f}")
        print(f"  Required: ${total_cost:,.2f}")
        
        if balance >= total_cost:
            print(f"  [OK] Sufficient balance")
            
            # 設置停損和停利
            stop_loss = offer * 0.95  # 5% 停損
            take_profit = offer * 1.10  # 10% 停利
            
            print(f"\n[5] Risk Management:")
            print(f"  Stop Loss: ${stop_loss:,.2f} (-5%)")
            print(f"  Take Profit: ${take_profit:,.2f} (+10%)")
            
            # 執行購買訂單
            print(f"\n[6] Placing BUY order...")
            result = system.place_order(
                epic="BTCUSD",
                direction="BUY",
                size=1.0,  # 1 BTC
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result['success']:
                print("\n" + "="*60)
                print(" ✓ ORDER EXECUTED SUCCESSFULLY")
                print("="*60)
                
                deal_ref = result['data'].get('dealReference')
                print(f"  Deal Reference: {deal_ref}")
                print(f"  Status: FILLED")
                print(f"  Type: BUY")
                print(f"  Size: 1.0 BTC")
                print(f"  Entry Price: ${offer:,.2f}")
                print(f"  Stop Loss: ${stop_loss:,.2f}")
                print(f"  Take Profit: ${take_profit:,.2f}")
                
                # 獲取更新後的持倉
                print(f"\n[7] Verifying position...")
                positions = system.collector.get_positions()
                
                btc_position = None
                for pos in positions:
                    if pos.get('market', {}).get('epic') == 'BTCUSD':
                        btc_position = pos
                        break
                
                if btc_position:
                    position_data = btc_position.get('position', {})
                    print(f"  Position confirmed:")
                    print(f"  Deal ID: {position_data.get('dealId')}")
                    print(f"  Size: {position_data.get('size')} BTC")
                    print(f"  Direction: {position_data.get('direction')}")
                    print(f"  Open Level: ${position_data.get('level'):,.2f}")
                
                # 計算預期盈虧
                print(f"\n[8] Potential Outcomes:")
                potential_profit = (take_profit - offer) * 1.0
                potential_loss = (offer - stop_loss) * 1.0
                print(f"  Maximum Profit: ${potential_profit:,.2f}")
                print(f"  Maximum Loss: ${potential_loss:,.2f}")
                
                print("\n" + "="*60)
                print(" Bitcoin purchase completed!")
                print(" Position will be monitored automatically")
                print("="*60)
                
                return True
                
            else:
                print("\n[ERROR] Order failed:")
                print(f"  Reason: {result.get('error')}")
                return False
        else:
            print(f"  [ERROR] Insufficient balance")
            print(f"  Need ${total_cost - balance:,.2f} more")
            return False
    else:
        print("[ERROR] Could not get Bitcoin market data")
        return False

if __name__ == "__main__":
    # 執行比特幣購買
    success = buy_bitcoin()
    
    if not success:
        print("\n[FAILED] Bitcoin purchase could not be completed")
        print("Please check your account settings and try again")