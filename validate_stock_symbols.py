#!/usr/bin/env python3
"""
驗證股票符號在Capital.com和Yahoo Finance的可用性
創建一個雙方都支援的股票列表
"""

import yfinance as yf
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import time

def load_capital_stocks():
    """載入Capital.com股票列表"""
    stocks = []
    
    # 從capital_real_stocks.json載入
    if os.path.exists('capital_real_stocks.json'):
        with open('capital_real_stocks.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                ticker = item.get('ticker', '')
                epic = item.get('epic', '')
                name = item.get('name', '')
                
                # 嘗試不同的符號格式
                symbols_to_try = []
                
                # 1. 原始ticker
                if ticker:
                    symbols_to_try.append(ticker)
                
                # 2. Epic可能包含更完整的符號
                if epic and epic != ticker:
                    # 移除特殊後綴
                    clean_epic = epic.replace('USD', '').replace('GBP', '').replace('EUR', '')
                    if clean_epic and clean_epic != ticker:
                        symbols_to_try.append(clean_epic)
                
                # 3. 如果是加密貨幣，加上-USD後綴
                crypto_keywords = ['BTC', 'ETH', 'DOGE', 'ADA', 'DOT', 'LINK', 'UNI', 'MATIC']
                if any(keyword in ticker.upper() for keyword in crypto_keywords):
                    symbols_to_try.append(f"{ticker}-USD")
                
                stocks.append({
                    'ticker': ticker,
                    'epic': epic,
                    'name': name,
                    'symbols_to_try': symbols_to_try
                })
    
    print(f"Loaded {len(stocks)} stocks from Capital.com")
    return stocks

def validate_yahoo_finance(symbol):
    """驗證符號在Yahoo Finance是否有效"""
    try:
        # 嘗試下載最近的數據
        data = yf.download(
            symbol,
            start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            end=datetime.now().strftime('%Y-%m-%d'),
            interval='1d',
            progress=False,
            threads=False
        )
        
        # 檢查是否有數據
        if not data.empty and len(data) > 5:
            return True, len(data)
        return False, 0
        
    except Exception as e:
        return False, 0

def find_valid_symbols():
    """找出在兩個平台都有效的符號"""
    print("="*80)
    print("VALIDATING STOCK SYMBOLS")
    print("="*80)
    
    # 載入Capital.com股票
    capital_stocks = load_capital_stocks()
    
    # 驗證結果
    valid_symbols = []
    invalid_symbols = []
    
    # 批量驗證
    print("\nValidating symbols on Yahoo Finance...")
    
    for stock in tqdm(capital_stocks[:1000], desc="Validating"):  # 先測試前1000個
        found = False
        valid_symbol = None
        
        # 嘗試不同的符號格式
        for symbol in stock['symbols_to_try']:
            is_valid, data_points = validate_yahoo_finance(symbol)
            
            if is_valid:
                found = True
                valid_symbol = symbol
                break
        
        if found:
            valid_symbols.append({
                'symbol': valid_symbol,
                'capital_ticker': stock['ticker'],
                'capital_epic': stock['epic'],
                'name': stock['name']
            })
        else:
            invalid_symbols.append(stock['ticker'])
        
        # 每50個符號暫停一下，避免被限制
        if len(valid_symbols) % 50 == 0 and len(valid_symbols) > 0:
            time.sleep(0.5)
    
    print(f"\n[RESULT] Valid symbols: {len(valid_symbols)}")
    print(f"[RESULT] Invalid symbols: {len(invalid_symbols)}")
    
    return valid_symbols

def save_validated_list(valid_symbols):
    """保存驗證過的符號列表"""
    
    # 1. 保存完整的映射表
    with open('validated_stocks_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(valid_symbols, f, indent=2, ensure_ascii=False)
    
    # 2. 保存Yahoo Finance符號列表（用於下載數據）
    with open('validated_yahoo_symbols.txt', 'w') as f:
        for item in valid_symbols:
            f.write(f"{item['symbol']}\n")
    
    # 3. 保存Capital.com符號列表（用於交易）
    with open('validated_capital_symbols.txt', 'w') as f:
        for item in valid_symbols:
            f.write(f"{item['capital_ticker']}\n")
    
    print(f"\n[SAVED] validated_stocks_mapping.json - Complete mapping")
    print(f"[SAVED] validated_yahoo_symbols.txt - Yahoo Finance symbols")
    print(f"[SAVED] validated_capital_symbols.txt - Capital.com symbols")

def add_reliable_stocks():
    """添加已知可靠的股票"""
    # 這些是已知在兩個平台都有效的主要股票
    reliable_stocks = [
        # 科技股
        {'symbol': 'AAPL', 'capital_ticker': 'AAPL', 'capital_epic': 'AAPL', 'name': 'Apple Inc'},
        {'symbol': 'MSFT', 'capital_ticker': 'MSFT', 'capital_epic': 'MSFT', 'name': 'Microsoft'},
        {'symbol': 'GOOGL', 'capital_ticker': 'GOOGL', 'capital_epic': 'GOOGL', 'name': 'Alphabet Inc'},
        {'symbol': 'AMZN', 'capital_ticker': 'AMZN', 'capital_epic': 'AMZN', 'name': 'Amazon'},
        {'symbol': 'META', 'capital_ticker': 'META', 'capital_epic': 'META', 'name': 'Meta Platforms'},
        {'symbol': 'NVDA', 'capital_ticker': 'NVDA', 'capital_epic': 'NVDA', 'name': 'NVIDIA'},
        {'symbol': 'TSLA', 'capital_ticker': 'TSLA', 'capital_epic': 'TSLA', 'name': 'Tesla'},
        {'symbol': 'AMD', 'capital_ticker': 'AMD', 'capital_epic': 'AMD', 'name': 'AMD'},
        {'symbol': 'INTC', 'capital_ticker': 'INTC', 'capital_epic': 'INTC', 'name': 'Intel'},
        {'symbol': 'NFLX', 'capital_ticker': 'NFLX', 'capital_epic': 'NFLX', 'name': 'Netflix'},
        
        # 金融股
        {'symbol': 'JPM', 'capital_ticker': 'JPM', 'capital_epic': 'JPM', 'name': 'JPMorgan Chase'},
        {'symbol': 'BAC', 'capital_ticker': 'BAC', 'capital_epic': 'BAC', 'name': 'Bank of America'},
        {'symbol': 'WFC', 'capital_ticker': 'WFC', 'capital_epic': 'WFC', 'name': 'Wells Fargo'},
        {'symbol': 'GS', 'capital_ticker': 'GS', 'capital_epic': 'GS', 'name': 'Goldman Sachs'},
        {'symbol': 'MS', 'capital_ticker': 'MS', 'capital_epic': 'MS', 'name': 'Morgan Stanley'},
        {'symbol': 'V', 'capital_ticker': 'V', 'capital_epic': 'V', 'name': 'Visa'},
        {'symbol': 'MA', 'capital_ticker': 'MA', 'capital_epic': 'MA', 'name': 'Mastercard'},
        {'symbol': 'AXP', 'capital_ticker': 'AXP', 'capital_epic': 'AXP', 'name': 'American Express'},
        
        # 醫療保健
        {'symbol': 'JNJ', 'capital_ticker': 'JNJ', 'capital_epic': 'JNJ', 'name': 'Johnson & Johnson'},
        {'symbol': 'UNH', 'capital_ticker': 'UNH', 'capital_epic': 'UNH', 'name': 'UnitedHealth'},
        {'symbol': 'PFE', 'capital_ticker': 'PFE', 'capital_epic': 'PFE', 'name': 'Pfizer'},
        {'symbol': 'ABBV', 'capital_ticker': 'ABBV', 'capital_epic': 'ABBV', 'name': 'AbbVie'},
        {'symbol': 'MRK', 'capital_ticker': 'MRK', 'capital_epic': 'MRK', 'name': 'Merck'},
        {'symbol': 'CVS', 'capital_ticker': 'CVS', 'capital_epic': 'CVS', 'name': 'CVS Health'},
        
        # 消費品
        {'symbol': 'WMT', 'capital_ticker': 'WMT', 'capital_epic': 'WMT', 'name': 'Walmart'},
        {'symbol': 'HD', 'capital_ticker': 'HD', 'capital_epic': 'HD', 'name': 'Home Depot'},
        {'symbol': 'PG', 'capital_ticker': 'PG', 'capital_epic': 'PG', 'name': 'Procter & Gamble'},
        {'symbol': 'KO', 'capital_ticker': 'KO', 'capital_epic': 'KO', 'name': 'Coca-Cola'},
        {'symbol': 'PEP', 'capital_ticker': 'PEP', 'capital_epic': 'PEP', 'name': 'PepsiCo'},
        {'symbol': 'COST', 'capital_ticker': 'COST', 'capital_epic': 'COST', 'name': 'Costco'},
        {'symbol': 'NKE', 'capital_ticker': 'NKE', 'capital_epic': 'NKE', 'name': 'Nike'},
        {'symbol': 'MCD', 'capital_ticker': 'MCD', 'capital_epic': 'MCD', 'name': 'McDonalds'},
        {'symbol': 'SBUX', 'capital_ticker': 'SBUX', 'capital_epic': 'SBUX', 'name': 'Starbucks'},
        
        # 能源
        {'symbol': 'XOM', 'capital_ticker': 'XOM', 'capital_epic': 'XOM', 'name': 'Exxon Mobil'},
        {'symbol': 'CVX', 'capital_ticker': 'CVX', 'capital_epic': 'CVX', 'name': 'Chevron'},
        {'symbol': 'COP', 'capital_ticker': 'COP', 'capital_epic': 'COP', 'name': 'ConocoPhillips'},
        
        # 工業
        {'symbol': 'BA', 'capital_ticker': 'BA', 'capital_epic': 'BA', 'name': 'Boeing'},
        {'symbol': 'CAT', 'capital_ticker': 'CAT', 'capital_epic': 'CAT', 'name': 'Caterpillar'},
        {'symbol': 'LMT', 'capital_ticker': 'LMT', 'capital_epic': 'LMT', 'name': 'Lockheed Martin'},
        {'symbol': 'UPS', 'capital_ticker': 'UPS', 'capital_epic': 'UPS', 'name': 'UPS'},
        {'symbol': 'FDX', 'capital_ticker': 'FDX', 'capital_epic': 'FDX', 'name': 'FedEx'},
        
        # 中概股
        {'symbol': 'BABA', 'capital_ticker': 'BABA', 'capital_epic': 'BABA', 'name': 'Alibaba'},
        {'symbol': 'JD', 'capital_ticker': 'JD', 'capital_epic': 'JD', 'name': 'JD.com'},
        {'symbol': 'PDD', 'capital_ticker': 'PDD', 'capital_epic': 'PDD', 'name': 'PDD Holdings'},
        {'symbol': 'BIDU', 'capital_ticker': 'BIDU', 'capital_epic': 'BIDU', 'name': 'Baidu'},
        {'symbol': 'NIO', 'capital_ticker': 'NIO', 'capital_epic': 'NIO', 'name': 'NIO Inc'},
        {'symbol': 'XPEV', 'capital_ticker': 'XPEV', 'capital_epic': 'XPEV', 'name': 'XPeng'},
        {'symbol': 'LI', 'capital_ticker': 'LI', 'capital_epic': 'LI', 'name': 'Li Auto'},
        
        # ETFs
        {'symbol': 'SPY', 'capital_ticker': 'SPY', 'capital_epic': 'SPY', 'name': 'SPDR S&P 500'},
        {'symbol': 'QQQ', 'capital_ticker': 'QQQ', 'capital_epic': 'QQQ', 'name': 'Invesco QQQ'},
        {'symbol': 'IWM', 'capital_ticker': 'IWM', 'capital_epic': 'IWM', 'name': 'iShares Russell 2000'},
        {'symbol': 'DIA', 'capital_ticker': 'DIA', 'capital_epic': 'DIA', 'name': 'SPDR Dow Jones'},
        {'symbol': 'VOO', 'capital_ticker': 'VOO', 'capital_epic': 'VOO', 'name': 'Vanguard S&P 500'},
        {'symbol': 'VTI', 'capital_ticker': 'VTI', 'capital_epic': 'VTI', 'name': 'Vanguard Total Market'},
    ]
    
    return reliable_stocks

def main():
    print("="*80)
    print("CREATING VALIDATED STOCK LIST")
    print("="*80)
    
    # 1. 先添加已知可靠的股票
    print("\n[STEP 1] Adding known reliable stocks...")
    reliable = add_reliable_stocks()
    print(f"Added {len(reliable)} reliable stocks")
    
    # 2. 驗證Capital.com列表中的其他股票
    print("\n[STEP 2] Validating Capital.com stocks...")
    validated = find_valid_symbols()
    
    # 3. 合併列表（去重）
    all_symbols = reliable.copy()
    seen_symbols = {s['symbol'] for s in reliable}
    
    for stock in validated:
        if stock['symbol'] not in seen_symbols:
            all_symbols.append(stock)
            seen_symbols.add(stock['symbol'])
    
    print(f"\n[FINAL] Total validated stocks: {len(all_symbols)}")
    
    # 4. 保存結果
    save_validated_list(all_symbols)
    
    # 5. 顯示樣本
    print("\n" + "="*80)
    print("SAMPLE VALIDATED STOCKS:")
    print("="*80)
    for stock in all_symbols[:20]:
        print(f"{stock['symbol']:10} -> {stock['capital_ticker']:10} : {stock['name']}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print(f"Total validated stocks: {len(all_symbols)}")
    print("Files created:")
    print("  - validated_stocks_mapping.json (complete mapping)")
    print("  - validated_yahoo_symbols.txt (for data download)")
    print("  - validated_capital_symbols.txt (for trading)")
    print("="*80)

if __name__ == "__main__":
    main()