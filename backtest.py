# backtest.py

import pandas as pd
from backtesting import Backtest
import warnings
import os
import sys
from datetime import datetime # --- 新增導入 datetime 模組 ---

# 假設您的 historical_data.py 和 strategies.py 檔案位於 src 資料夾下
# backtest.py 與 src 在同一層級 (專案根目錄)，所以使用以下導入方式
try:
    from src.historical_data import download_data 
    from src.strategies import MaCrossoverSlTpPercent 
except ImportError as e:
    print(f"匯入模組時出錯: {e}")
    print("請確認您是從專案根目錄執行此腳本，並且 src 資料夾中有 historical_data.py 和 strategies.py 檔案。")
    exit()

# --- 1. 設定回測參數 ---
# 數據下載參數
ticker_symbol = "BTC-USD"           # 您想回測的股票/商品代號
start_date_str = "2021-01-01"       # 回測開始日期

# --- 以下是修正的部分 ---
# 動態獲取今天的日期作為結束日期，以確保數據總是最新的
end_date_str = datetime.today().strftime('%Y-%m-%d')

# 回測設定參數
initial_cash = 100000               # 初始資金
commission_rate = 0.0002            # 佣金率 (例如 0.02%)

# --- 2. 獲取歷史數據 ---
# 直接從 yfinance 下載數據
ohlcv_data = download_data(ticker=ticker_symbol, 
                           start_date=start_date_str, 
                           end_date=end_date_str)

# --- 3. 執行回測與分析 ---
if ohlcv_data is not None and not ohlcv_data.empty:
    print(f"\n準備對 {ticker_symbol} 執行策略回測...")
    
    # 實例化 Backtest 物件
    bt = Backtest(ohlcv_data, 
                  MaCrossoverSlTpPercent, 
                  cash=initial_cash, 
                  commission=commission_rate)

    # 執行回測
    print("執行回測中...")
    stats = bt.run()

    # 印出回測統計數據
    with open("backtest_result.txt", "w", encoding="utf-8") as f:
        f.write("--- 回測結果統計 ---\n")
        f.write(str(stats))
    print("回測結果已寫入 backtest_result.txt")

    # 查看特定統計數據 (如果回測成功)
    if '# Trades' in stats and stats['# Trades'] > 0:
        print(f"\n夏普比率 (Sharpe Ratio): {stats['Sharpe Ratio']:.2f}")
        print(f"最大回撤 (Max. Drawdown [%]): {stats['Max. Drawdown [%]']:.2f}")
        print(f"勝率 (Win Rate [%]): {stats['Win Rate [%]']:.2f}")
        print(f"總交易次數: {stats['# Trades']}")
    else:
        print("\n回測期間內沒有產生任何交易。")

    # 繪製回測結果圖表
    print("\n產生回測結果圖表...")
    try:
        # 暫時關閉 stderr
        stderr_fileno = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        bt.plot(open_browser=True)
        sys.stderr.close()
        sys.stderr = stderr_fileno
    except Exception as e_plot:
        print(f"繪製圖表時發生錯誤 (可能在無圖形介面環境): {e_plot}")
        print("您可以嘗試 bt.plot(open_browser=False, filename='backtest_plot.html') 來儲存圖表。")

    # --- 4. (可選) 參數優化範例 ---
    # print("\n--- 執行參數優化 (範例，可能耗時較長) ---")
    # stats_optimized = bt.optimize(
    #     ma_short_len=range(5, 11, 1),
    #     ma_long_len=range(20, 31, 2),
    #     maximize='Sharpe Ratio',
    #     constraint=lambda param: param.ma_short_len < param.ma_long_len,
    #     max_tries=50
    # )
    # print("\n--- 參數優化結果 ---")
    # print(stats_optimized)
    # print("\n最佳參數組合:")
    # print(stats_optimized._strategy)

else:
    print("數據載入失敗，無法執行回測。")
    