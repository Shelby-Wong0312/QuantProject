#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新所有Agent使用Capital.com取代MT4
"""

import os
import glob

def update_agents():
    """更新所有agent文件"""
    
    print("\n" + "="*60)
    print(" Updating All Agents to Use Capital.com")
    print("="*60)
    
    # Agent文件列表
    agent_files = glob.glob('.cloud/*_agent.md')
    
    # 要替換的內容
    replacements = [
        ("MT4", "Capital.com"),
        ("ZeroMQ", "REST API"),
        ("DWX", "Capital.com API"),
        ("Expert Advisor", "Trading System"),
        ("MQL4", "Python"),
        ("mt4_", "capital_"),
        ("MT4 bridge", "Capital.com connector"),
        ("32768", "HTTPS"),
        ("32769", "WebSocket"),
        ("CRUDEOIL", "OIL_CRUDE"),
    ]
    
    updated_count = 0
    
    for agent_file in agent_files:
        print(f"\nUpdating {os.path.basename(agent_file)}...")
        
        try:
            # 讀取文件
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 執行替換
            for old, new in replacements:
                content = content.replace(old, new)
            
            # 如果內容有變化，寫回文件
            if content != original_content:
                with open(agent_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  [UPDATED] Changes applied")
                updated_count += 1
            else:
                print(f"  [SKIP] No changes needed")
                
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    # 創建Capital.com專用Agent
    capital_agent_content = """# Capital.com Trading Agent

## Role
Specialized agent for Capital.com API integration and trading execution.

## Responsibilities
1. **API Management**
   - Maintain Capital.com REST API connections
   - Handle authentication and session management
   - Monitor API rate limits
   - Implement WebSocket for real-time data

2. **Trade Execution**
   - Execute market orders (BUY/SELL)
   - Manage stop-loss and take-profit orders
   - Handle position sizing and risk management
   - Monitor order status and fills

3. **Account Management**
   - Track account balance and equity
   - Monitor margin requirements
   - Calculate position sizes
   - Track P&L in real-time

4. **Market Data**
   - Stream real-time prices
   - Collect historical OHLCV data
   - Monitor multiple markets simultaneously
   - Calculate technical indicators

## Technical Stack
- **API**: Capital.com REST API v1
- **Languages**: Python 3.9+
- **Libraries**: requests, pandas, numpy
- **Storage**: SQLite, JSON

## Current Status
### Active Positions
- BTCUSD: 1.0 BTC @ $116,465.30 (P&L: +$0.70)

### Account Status
- Balance: $137,766.45
- Available: $131,942.90
- Total P&L: +$5.65

### Available Markets (29)
- Crypto: BTCUSD, ETHUSD, LTCBTC
- Forex: EURUSD, GBPUSD, USDJPY
- Commodities: GOLD, OIL_CRUDE, SILVER
- Indices: US100, DE40, SP35

## Key Files
- `capital_data_collector.py` - Data collection
- `capital_trading_system.py` - Trading execution
- `capital_live_trading.py` - Automated trading
- `buy_bitcoin_now.py` - Direct order execution
- `check_bitcoin_position.py` - Position monitoring

## Integration Points
- Provides execution for **Quant Agent** strategies
- Reports trades to **PM Agent**
- Coordinates with **Risk Agent** for position sizing
- Sends data to **DE Agent** for processing
"""
    
    with open('.cloud/capital_agent.md', 'w', encoding='utf-8') as f:
        f.write(capital_agent_content)
    print(f"\nCreated new Capital.com Agent")
    
    print("\n" + "="*60)
    print(" Update Summary")
    print("="*60)
    print(f" Total Agents: {len(agent_files)}")
    print(f" Updated: {updated_count}")
    print(f" New Agent: capital_agent.md")
    print("="*60)

if __name__ == "__main__":
    update_agents()