# src/historical_data.py

import yfinance as yf
import pandas as pd

def download_data(ticker="BTC-USD", start_date="2020-01-01", end_date="2023-12-31", filename_to_save=None):
    """
    下載指定股票代號的歷史 OHLCV 數據。

    參數:
    ticker (str): 股票代號 (例如 "BTC-USD", "AAPL").
    start_date (str): 開始日期 (格式 "YYYY-MM-DD").
    end_date (str): 結束日期 (格式 "YYYY-MM-DD").
    filename_to_save (str, optional): 如果提供，則將數據儲存到此 CSV 檔名. 預設為 None (不儲存).

    返回:
    pandas.DataFrame: 包含 OHLCV 數據的 DataFrame，如果下載失敗則返回 None.
    """
    print(f"開始下載 {ticker} 從 {start_date} 到 {end_date} 的數據...")
    try:
        data_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data_df.empty:
            print(f"在指定日期範圍內沒有下載到 {ticker} 的數據。")
            return None

        # --- 新增的修正部分 ---
        # 檢查欄位是否為 MultiIndex，如果是，則進行扁平化
        if isinstance(data_df.columns, pd.MultiIndex):
            # 將多層級索引 (例如 ('Open', 'BTC-USD')) 轉換為單層級索引 (例如 'Open')
            data_df.columns = data_df.columns.get_level_values(0)
            print("偵測到 MultiIndex 欄位，已進行扁平化處理。")
        # --- 修正部分結束 ---

        # 確保欄位名符合 Backtesting.py 要求 (Open, High, Low, Close, Volume)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data_df.columns for col in required_columns):
            print(f"下載的 {ticker} 數據缺少必要的欄位。需要的欄位: {required_columns}，實際欄位: {data_df.columns.tolist()}")
            return None

        print(f"成功下載 {len(data_df)} 筆 {ticker} 的數據。")
        
        if filename_to_save:
            try:
                data_df.to_csv(filename_to_save)
                print(f"數據已儲存到 {filename_to_save}")
            except Exception as e_save:
                print(f"儲存數據到 {filename_to_save} 時發生錯誤: {e_save}")
                
        return data_df
        
    except Exception as e_download:
        print(f"下載 {ticker} 數據時發生錯誤: {e_download}")
        return None

if __name__ == '__main__':
    # --- 測試下載 BTC-USD ---
    btc_ticker = "BTC-USD"
    btc_start = "2021-01-01"
    btc_end = "2023-12-31"
    btc_filename = f"{btc_ticker}_data_{btc_start}_to_{btc_end}.csv"
    
    btc_data = download_data(ticker=btc_ticker, 
                             start_date=btc_start, 
                             end_date=btc_end, 
                             filename_to_save=btc_filename)
    
    if btc_data is not None:
        print(f"\n{btc_ticker} 數據概覽:")
        print(btc_data.head())
        print(f"\n{btc_ticker} 數據的欄位:")
        print(btc_data.columns) # 檢查欄位是否已變為單層級