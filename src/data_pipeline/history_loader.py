# quant_project/data/history_loader.py
# ALPACA VERSION

import logging
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import config

logger = logging.getLogger(__name__)


class HistoryLoader:
    """
    使用 Alpaca API 獲取歷史K線數據。
    """

    def __init__(self):
        self.api = REST(
            key_id=config.ALPACA_API_KEY_ID,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url="https://paper-api.alpaca.markets",
            api_version="v2",
        )
        logger.info("Alpaca 歷史數據加載器已初始化。")

    def get_bars(
        self, symbol: str, timeframe: TimeFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        獲取指定時間範圍內的歷史K線數據。
        :param symbol: 股票代碼, e.g., "AAPL"
        :param timeframe: 時間週期, e.g., TimeFrame.Day, TimeFrame.Hour
        :param start_date: 開始日期, "YYYY-MM-DD"
        :param end_date: 結束日期, "YYYY-MM-DD"
        :return: 包含 OHLCV 數據的 DataFrame
        """
        try:
            bars_df = self.api.get_bars(
                symbol, timeframe, start=start_date, end=end_date, adjustment="raw"
            ).df

            # Alpaca返回的DataFrame欄位名是小寫，我們將其轉換為標準的大寫
            bars_df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )

            logger.info(
                f"成功獲取 {symbol} 從 {start_date} 到 {end_date} 的 {len(bars_df)} 筆數據。"
            )
            return bars_df

        except Exception as e:
            logger.error(f"獲取 {symbol} 歷史數據時出錯: {e}")
            return pd.DataFrame()
