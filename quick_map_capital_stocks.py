#!/usr/bin/env python3
"""
快速映射Capital.com核心股票到Yahoo Finance
優先處理最活躍和最重要的股票
"""

import json
import yfinance as yf
from datetime import datetime, timedelta
import time

def get_priority_capital_stocks():
    """獲取優先級高的Capital.com股票"""
    
    # 載入所有Capital.com股票
    all_stocks = []
    if os.path.exists('capital_real_stocks.json'):
        with open('capital_real_stocks.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_stocks = data
    
    # 優先處理的股票（這些是最常交易的）
    priority_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL',
        'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK',
        'WMT', 'HD', 'KO', 'PEP', 'NKE', 'MCD', 'SBUX',
        'XOM', 'CVX', 'COP', 'SLB',
        'BA', 'CAT', 'LMT', 'UPS', 'FDX',
        'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'
    ]
    
    # 篩選出優先股票
    priority_stocks = []
    other_stocks = []
    
    for stock in all_stocks:
        ticker = stock.get('ticker', '')
        if ticker in priority_tickers or any(p in ticker for p in priority_tickers):
            priority_stocks.append(stock)
        else:
            other_stocks.append(stock)
    
    # 返回優先股票 + 部分其他股票
    return priority_stocks + other_stocks[:500]  # 總共處理約550個股票

def validate_yahoo(symbol):
    """快速驗證Yahoo Finance符號"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        return not hist.empty
    except:
        return False

def map_stock(capital_stock):
    """映射單個股票"""
    ticker = capital_stock.get('ticker', '')
    epic = capital_stock.get('epic', '')
    
    # 嘗試的符號順序
    candidates = []
    
    # 1. 直接使用ticker
    if ticker:
        candidates.append(ticker)
    
    # 2. 清理ticker（移除USD等後綴）
    cleaned = ticker
    for suffix in ['USD', 'GBP', 'EUR', 'JPY', 'CHF', 'CAD']:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            candidates.append(cleaned)
    
    # 3. 嘗試epic
    if epic and epic != ticker:
        candidates.append(epic)
    
    # 4. 特殊處理加密貨幣
    crypto_keywords = ['BTC', 'ETH', 'DOGE', 'ADA', 'DOT', 'LINK']
    if any(k in ticker.upper() for k in crypto_keywords):
        candidates.append(f"{ticker}-USD")
    
    # 測試每個候選
    for candidate in candidates:
        if validate_yahoo(candidate):
            return candidate
    
    return None

def main():
    print("="*80)
    print("QUICK CAPITAL.COM TO YAHOO FINANCE MAPPING")
    print("="*80)
    
    # 獲取要處理的股票
    stocks_to_map = get_priority_capital_stocks()
    print(f"[INFO] Processing {len(stocks_to_map)} priority Capital.com stocks")
    
    # 映射
    mapped = []
    unmapped = []
    
    print("\n[MAPPING] Starting quick mapping...")
    for i, stock in enumerate(stocks_to_map):
        yahoo_symbol = map_stock(stock)
        
        if yahoo_symbol:
            mapped.append({
                'capital': stock.get('ticker', ''),
                'yahoo': yahoo_symbol,
                'name': stock.get('name', '')
            })
        else:
            unmapped.append(stock.get('ticker', ''))
        
        # 進度顯示
        if (i + 1) % 50 == 0:
            print(f"[PROGRESS] Processed {i+1}/{len(stocks_to_map)} - Mapped: {len(mapped)}")
            time.sleep(0.5)  # 避免請求過快
    
    # 保存結果
    print(f"\n[RESULT] Successfully mapped: {len(mapped)} stocks")
    print(f"[RESULT] Failed to map: {len(unmapped)} stocks")
    
    # 保存映射文件
    with open('quick_capital_yahoo_map.json', 'w', encoding='utf-8') as f:
        json.dump(mapped, f, indent=2, ensure_ascii=False)
    
    # 保存Yahoo符號列表
    with open('quick_yahoo_symbols.txt', 'w') as f:
        for item in mapped:
            f.write(f"{item['yahoo']}\n")
    
    # 保存Capital符號列表
    with open('quick_capital_symbols.txt', 'w') as f:
        for item in mapped:
            f.write(f"{item['capital']}\n")
    
    print("\n[SAVED] Files created:")
    print("  - quick_capital_yahoo_map.json")
    print("  - quick_yahoo_symbols.txt")
    print("  - quick_capital_symbols.txt")
    
    # 顯示樣本
    print("\nSample mappings:")
    for item in mapped[:20]:
        print(f"  {item['capital']:10} -> {item['yahoo']:10} : {item['name'][:30]}")
    
    print("\n" + "="*80)
    print(f"QUICK MAPPING COMPLETE! {len(mapped)} stocks ready for use")
    print("="*80)

if __name__ == "__main__":
    import os
    main()