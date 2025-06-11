# backtesting_scripts/test_single_stock.py

import sys
import os
# 將專案根目錄加入 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtesting import Backtest
import yfinance as yf
from datetime import datetime

from adapters.backtesting_adapter import BacktestingPyAdapter
from strategy.concrete_strategies.confluence_strategy import ConfluenceStrategy

# 測試一個知名的股票
TICKER = "AAPL"  # 蘋果公司

print(f"開始下載 {TICKER} 的歷史數據...")
data_df = yf.download(TICKER, start="2021-01-01", end=datetime.today().strftime('%Y-%m-%d'), interval="1d")

if data_df.empty:
    print(f"無法下載 {TICKER} 的數據")
    exit(1)

# 處理 yfinance 的多層列索引問題
if isinstance(data_df.columns, pd.MultiIndex):
    data_df.columns = data_df.columns.get_level_values(0)
    
data_df.columns = [col.capitalize() for col in data_df.columns]
print(f"成功載入 {len(data_df)} 筆 {TICKER} 數據。")

# 設定策略參數
strategy_parameters = {
    'symbol': TICKER,
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

# 初始化回測
bt = Backtest(
    data_df, 
    BacktestingPyAdapter, 
    cash=100000,  # 10萬美元初始資金
    commission=.002,
    margin=1.0
)

# 執行回測
print("\n開始執行回測...")
stats = bt.run(
    abstract_strategy_class=ConfluenceStrategy,
    strategy_params=strategy_parameters
)

# 顯示結果
print("\n" + "="*50)
print(f"          {TICKER} 回測結果")
print("="*50)
print(f"總報酬率: {stats['Return [%]']:.2f}%")
print(f"夏普比率: {stats['Sharpe Ratio']:.2f}")
print(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%")
print(f"勝率: {stats['Win Rate [%]']:.2f}%")
print(f"交易次數: {stats['# Trades']}")

# 顯示詳細交易記錄
if stats['_trades'] is not None and not stats['_trades'].empty:
    print("\n交易記錄:")
    print(stats['_trades'][['Size', 'EntryPrice', 'ExitPrice', 'ReturnPct', 'Duration']])

# 繪製圖表
print("\n正在生成圖表...")
bt.plot() 