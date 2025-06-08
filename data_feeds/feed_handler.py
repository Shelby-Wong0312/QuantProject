# 檔案位置: data_feeds/feed_handler.py

import pandas as pd
import time
import logging
from core.event import MarketDataEvent

logger = logging.getLogger(__name__)

class HistoricalDataFeedHandler:
    """
    一個從 CSV 檔案讀取歷史數據，並模擬即時數據流的事件產生器。
    """
    def __init__(self, event_queue, csv_filepath: str, symbol: str, interval_seconds: float = 0.1):
        """
        初始化 Feed Handler。
        :param event_queue: 事件隊列的實例。
        :param csv_filepath: 歷史數據 CSV 檔案的路徑。
        :param symbol: 交易標的代碼。
        :param interval_seconds: 模擬每根K棒之間的間隔時間（秒）。
        """
        self.event_queue = event_queue
        self.csv_filepath = csv_filepath
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        try:
            self.historical_data = pd.read_csv(self.csv_filepath, index_col=0, parse_dates=True)
            logger.info(f"成功從 {csv_filepath} 載入 {len(self.historical_data)} 筆原始數據。")

            # --- 以下是新增的數據清洗步驟 ---
            # 確保 OHLCV 欄位是數值類型，以防檔案中有文字字元
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in ohlcv_cols:
                if col in self.historical_data.columns:
                    # errors='coerce' 會將無法轉換的文字變成無效值 (NaN)
                    self.historical_data[col] = pd.to_numeric(self.historical_data[col], errors='coerce')
            
            # 移除任何因轉換錯誤產生的、含有 NaN 的數據行
            self.historical_data.dropna(inplace=True)
            logger.info(f"數據類型轉換與清洗完畢，剩餘有效數據 {len(self.historical_data)} 筆。")
            # --- 新增結束 ---

        except FileNotFoundError:
            logger.error(f"找不到歷史數據檔案: {self.csv_filepath}，請先運行回測腳本生成該檔案。")
            self.historical_data = None

    def start_feed(self):
        """開始回放數據，產生市場事件。"""
        if self.historical_data is None or self.historical_data.empty:
            logger.error("沒有數據可供回放，數據供給終止。")
            return

        logger.info(f"開始以每 {self.interval_seconds} 秒一筆的速度，回放歷史數據...")
        
        all_data_slice = pd.DataFrame()

        for index, row in self.historical_data.iterrows():
            current_bar_df = pd.DataFrame(row).T
            all_data_slice = pd.concat([all_data_slice, current_bar_df])
            
            event = MarketDataEvent(
                symbol=self.symbol,
                ohlcv_data=all_data_slice
            )
            
            self.event_queue.put(event)
            
            time.sleep(self.interval_seconds)
        
        logger.info("歷史數據回放完畢。")
        