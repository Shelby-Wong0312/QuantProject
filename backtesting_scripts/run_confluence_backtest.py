# backtesting_scripts/run_confluence_backtest.py

import pandas as pd
from backtesting import Backtest
import yfinance as yf
from datetime import datetime
import os

from adapters.backtesting_adapter import BacktestingPyAdapter
# 導入我們新的策略
from strategy.concrete_strategies.confluence_strategy import ConfluenceStrategy

# --- 1. 載入數據 ---
def load_symbols_from_file(filepath="tickers.txt"):
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
# 您可以修改這個變數來回測 tickers.txt 中的不同股票
all_symbols = load_symbols_from_file()
TICKER_TO_TEST = all_symbols[0] # 預設回測列表中的第一個

print(f"開始下載 {TICKER_TO_TEST} 的歷史數據...")
today_str = datetime.today().strftime('%Y-%m-%d')
data_df = yf.download(TICKER_TO_TEST, start="2021-01-01", end=today_str, interval="1d")

if data_df.empty:
    raise ValueError("無法下載數據，請檢查 ticker 或網路連線。")
    
data_df.columns = [col.capitalize() for col in data_df.columns]
print(f"成功載入 {len(data_df)} 筆 {TICKER_TO_TEST} 數據。")

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
bt = Backtest(
    data_df, 
    BacktestingPyAdapter, 
    cash=1000, # 初始資金
    commission=.002,
    margin=1.0
)

# --- 4. 執行回測 ---
stats = bt.run(
    abstract_strategy_class=ConfluenceStrategy,
    strategy_params=strategy_parameters
)

# --- 5. 優化輸出結果 ---
print("\n" + "="*50)
print(" " * 18 + "詳細交易紀錄")
print("="*50)
print(stats['_trades'])

print("\n" + "="*50)
print(" " * 19 + "整體回測績效")
print("="*50)
print(stats)

print("\n正在啟動瀏覽器以顯示圖表...")
bt.plot()