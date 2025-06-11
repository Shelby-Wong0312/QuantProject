# backtesting_scripts/run_confluence_backtest.py

import sys
import os
# 將專案根目錄加入 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtesting import Backtest
import yfinance as yf
from datetime import datetime

from adapters.backtesting_adapter import BacktestingPyAdapter
# 導入我們新的策略
from strategy.concrete_strategies.confluence_strategy import ConfluenceStrategy

# --- 1. 載入數據 ---
def load_symbols_from_file(filepath="valid_tickers.txt"):
    if not os.path.exists(filepath):
        print(f"Ticker file not found at {filepath}. Using default: ['BTC-USD']")
        return ["BTC-USD"]
    with open(filepath, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    if not symbols:
        print(f"Ticker file is empty. Using default: ['BTC-USD']")
        return ["BTC-USD"]
    return symbols

# 我們一次只回測一個標的，以利於觀察和分析
# 您可以修改這個變數來回測 valid_tickers.txt 中的不同股票
all_symbols = load_symbols_from_file()

# 選擇要測試的股票
# 可以選擇單一股票或批量測試
SINGLE_TEST = True  # 設為 False 來批量測試所有股票

# 一些知名股票的快速選項（如果您想測試特定股票）
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.'
}

# 選擇測試模式
USE_POPULAR_STOCK = True  # 設為 True 使用知名股票，False 使用 valid_tickers.txt
POPULAR_STOCK_SYMBOL = 'AAPL'  # 從 POPULAR_STOCKS 中選擇一個
TICKER_INDEX = 100    # 如果 USE_POPULAR_STOCK=False，選擇 valid_tickers.txt 中的股票索引

if SINGLE_TEST:
    # 單一股票測試
    if USE_POPULAR_STOCK:
        TICKER_TO_TEST = POPULAR_STOCK_SYMBOL
        print(f"使用知名股票: {TICKER_TO_TEST} - {POPULAR_STOCKS.get(TICKER_TO_TEST, '')}")
    else:
        if TICKER_INDEX >= len(all_symbols):
            print(f"索引 {TICKER_INDEX} 超出範圍，使用第一個股票")
            TICKER_INDEX = 0
        TICKER_TO_TEST = all_symbols[TICKER_INDEX]
        print(f"使用 valid_tickers.txt 中的第 {TICKER_INDEX} 個股票: {TICKER_TO_TEST}")
    
    print(f"開始下載 {TICKER_TO_TEST} 的歷史數據...")
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    # 下載股票數據
    data_df = yf.download(TICKER_TO_TEST, start="2021-01-01", end=today_str, interval="1d")
    
    if data_df.empty:
        print(f"無法下載 {TICKER_TO_TEST} 的數據，可能是無效的股票代碼")
        exit(1)
    
    # 處理 yfinance 的多層列索引問題
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = data_df.columns.get_level_values(0)
        
    data_df.columns = [col.capitalize() for col in data_df.columns]
    print(f"成功載入 {len(data_df)} 筆 {TICKER_TO_TEST} 數據。")
    
    # 檢查數據是否足夠
    if len(data_df) < 100:
        print(f"警告：{TICKER_TO_TEST} 的數據量較少（{len(data_df)} 筆），可能影響回測結果")

    # --- 2. 設定回測與策略參數 ---
    # 這裡定義的參數會傳入策略中，覆蓋策略檔案裡的預設值
    strategy_parameters = {
        'symbol': TICKER_TO_TEST,
        'short_ma_period': 5,
        'long_ma_period': 20,
        'bias_period': 20,
        'bias_upper': 7.0,
        'bias_lower': -8.0,
        'kd_k': 14, 'kd_d': 3, 'kd_smooth': 3,
        'kd_overbought': 80, 'kd_oversold': 20,
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
        'bb_period': 20, 'bb_std': 2.0,
        'ichi_tenkan': 9, 'ichi_kijun': 26, 'ichi_senkou_b': 52,
        'vol_ma_period': 20, 'vol_multiplier': 1.5,
        'atr_period': 14, 'atr_multiplier_sl': 2.0, 'atr_multiplier_tp': 4.0,
    }

    # --- 3. 初始化 Backtest 物件 ---
    # 根據股票價格調整初始資金
    initial_cash = 100000  # 預設 10 萬美元
    if len(data_df) > 0:
        avg_price = data_df['Close'].mean()
        if avg_price > 1000:  # 高價股（如 GOOGL, AMZN）
            initial_cash = 1000000  # 使用 100 萬美元
        elif avg_price < 10:  # 低價股
            initial_cash = 10000   # 使用 1 萬美元

    bt = Backtest(
        data_df, 
        BacktestingPyAdapter, 
        cash=initial_cash,
        commission=.002,  # 0.2% 手續費
        margin=1.0
    )

    # --- 4. 執行回測 ---
    stats = bt.run(
        abstract_strategy_class=ConfluenceStrategy,
        strategy_params=strategy_parameters
    )

    # --- 5. 優化輸出結果 ---
    print("\n" + "="*50)
    print(f"          {TICKER_TO_TEST} 詳細交易紀錄")
    print("="*50)
    if stats['_trades'] is not None and not stats['_trades'].empty:
        print(stats['_trades'])
    else:
        print("沒有產生任何交易")

    print("\n" + "="*50)
    print(f"           {TICKER_TO_TEST} 整體回測績效")
    print("="*50)
    print(stats)

    print("\n正在啟動瀏覽器以顯示圖表...")
    bt.plot()

else:
    # 批量測試模式
    print(f"開始批量測試 {len(all_symbols)} 個股票...")
    results = []
    
    for i, ticker in enumerate(all_symbols[:10]):  # 先測試前 10 個
        print(f"\n[{i+1}/10] 測試 {ticker}...")
        try:
            data_df = yf.download(ticker, start="2021-01-01", end=datetime.today().strftime('%Y-%m-%d'), 
                                interval="1d", progress=False)
            
            if data_df.empty or len(data_df) < 100:
                print(f"  跳過 {ticker}：數據不足")
                continue
                
            if isinstance(data_df.columns, pd.MultiIndex):
                data_df.columns = data_df.columns.get_level_values(0)
            data_df.columns = [col.capitalize() for col in data_df.columns]
            
            # 動態調整初始資金
            avg_price = data_df['Close'].mean()
            if avg_price > 1000:
                initial_cash = 1000000
            elif avg_price < 10:
                initial_cash = 10000
            else:
                initial_cash = 100000
                
            bt = Backtest(data_df, BacktestingPyAdapter, cash=initial_cash, commission=.002)
            
            # 需要為每個股票設定參數
            strategy_parameters = {
                'symbol': ticker,
                'short_ma_period': 5,
                'long_ma_period': 20,
                'bias_period': 20,
                'bias_upper': 7.0,
                'bias_lower': -8.0,
                'kd_k': 14, 'kd_d': 3, 'kd_smooth': 3,
                'kd_overbought': 80, 'kd_oversold': 20,
                'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
                'bb_period': 20, 'bb_std': 2.0,
                'ichi_tenkan': 9, 'ichi_kijun': 26, 'ichi_senkou_b': 52,
                'vol_ma_period': 20, 'vol_multiplier': 1.5,
                'atr_period': 14, 'atr_multiplier_sl': 2.0, 'atr_multiplier_tp': 4.0,
            }
            
            stats = bt.run(abstract_strategy_class=ConfluenceStrategy, strategy_params=strategy_parameters)
            
            # 收集關鍵指標
            results.append({
                'Symbol': ticker,
                'Return %': stats['Return [%]'],
                'Sharpe Ratio': stats['Sharpe Ratio'],
                'Max Drawdown %': stats['Max. Drawdown [%]'],
                'Win Rate %': stats['Win Rate [%]'],
                'Trades': stats['# Trades']
            })
            
        except Exception as e:
            print(f"  錯誤 {ticker}: {str(e)}")
            continue
    
    # 顯示批量測試結果
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Return %', ascending=False)
        
        print("\n" + "="*70)
        print("                    批量測試結果摘要")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # 儲存結果
        results_df.to_csv('confluence_backtest_results.csv', index=False)
        print(f"\n結果已儲存至 confluence_backtest_results.csv")