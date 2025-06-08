# 檔案位置: backtesting_scripts/run_rsi_ma_kd_backtest.py

import pandas as pd
from backtesting import Backtest
import yfinance as yf
from datetime import datetime

# 根據您的路徑，使用 strategy (單數)
from adapters.backtesting_adapter import BacktestingPyAdapter
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy

# --- 1. 載入數據 ---
try:
    # 動態獲取今天的日期作為結束日期，以繞過快取問題
    today_str = datetime.today().strftime('%Y-%m-%d')
    data_df = yf.download("BTC-USD", start="2021-01-01", end=today_str, interval="1d")
    
    if data_df.empty:
        raise ValueError("無法下載數據，請檢查 ticker 或網路連線。")
    
    # --- 以下是新增的部分 ---
    # 將下載的數據儲存為 CSV 檔案，以供下一階段使用
    data_df.to_csv('btc_usd_daily.csv', index_label='Date')
    print(f"成功載入 {len(data_df)} 筆 BTC-USD 數據，並已儲存至 btc_usd_daily.csv。")
    # --- 新增結束 ---

except Exception as e:
    print(f"數據下載失敗: {e}")
    exit()

# --- 數據清洗 ---
if isinstance(data_df.columns, pd.MultiIndex):
    data_df.columns = data_df.columns.droplevel(1)
data_df.columns = [col.capitalize() for col in data_df.columns]

# --- 2. 設定回測與策略參數 ---
strategy_parameters = {
    'symbol': 'BTC-USD',
    'short_ma_period': 10,
    'long_ma_period': 30,
    'rsi_period': 14,
    'rsi_oversold': 30,
}

# --- 3. 初始化 Backtest 物件 ---
bt = Backtest(
    data_df,
    BacktestingPyAdapter,
    cash=100_000,
    commission=.002,
    margin=1.0
)

# --- 4. 執行回測 ---
stats = bt.run(
    abstract_strategy_class=AbstractEnhancedRsiMaKdStrategy,
    strategy_params=strategy_parameters
)

# --- 5. 優化輸出結果 ---
print("\n" + "="*50)
print(" " * 18 + "詳細交易紀錄")
print("="*50)
trades_df = stats['_trades']
print(trades_df)

print("\n" + "="*50)
print(" " * 18 + "年度交易成果分析")
print("="*50)
if not trades_df.empty:
    trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'])
    yearly_stats = trades_df.groupby(trades_df['ExitTime'].dt.year)

    for year, group in yearly_stats:
        print(f"\n年份: {year}")
        print(f"----------------------")
        print(f"  總交易筆數: {len(group)}")
        print(f"  毛利 (PnL): {group['PnL'].sum():.2f}")
        
        win_trades = group[group['PnL'] > 0]
        loss_trades = group[group['PnL'] <= 0]
        win_rate = (len(win_trades) / len(group) * 100) if not group.empty else 0
        
        print(f"  勝率: {win_rate:.2f}%")
        if not win_trades.empty:
            print(f"  平均獲利: {win_trades['PnL'].mean():.2f}")
        if not loss_trades.empty:
            print(f"  平均虧損: {loss_trades['PnL'].mean():.2f}")
else:
    print("回測期間無任何交易。")

print("\n" + "="*50)
print(" " * 19 + "整體回測績效")
print("="*50)
print(stats)

print("\n正在啟動瀏覽器以顯示圖表...")
bt.plot()
