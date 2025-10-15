# quant_project/data/capital_history_loader.py
# Capital.com歷史數據加載器

import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys
import os

# 添加父目錄到Python路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class CapitalHistoryLoader:
    """
    使用 Capital.com API 獲取歷史K線數據。
    """

    def __init__(self):
        self.api_key = config.CAPITAL_API_KEY
        self.identifier = config.CAPITAL_IDENTIFIER
        self.password = config.CAPITAL_API_PASSWORD
        self.base_url = config.CAPITAL_BASE_URL
        self.cst = None
        self.x_security_token = None
        self.session = requests.Session()
        self._login()

    def _login(self):
        """登錄Capital.com API"""
        login_url = f"{self.base_url}/session"
        headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"identifier": self.identifier, "password": self.password}

        try:
            response = self.session.post(login_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                logger.info("成功登錄 Capital.com API")
            else:
                logger.error(f"登錄失敗: {response.status_code} - {response.text}")
                raise Exception("Capital.com API 登錄失敗")
        except requests.exceptions.RequestException as e:
            logger.error(f"登錄時發生網路錯誤: {e}")
            raise

    def get_bars(
        self, symbol: str, resolution: str, start_date: str, end_date: str, max_results: int = 1000
    ) -> pd.DataFrame:
        """
        獲取指定時間範圍內的歷史K線數據。

        :param symbol: 股票代碼，例如 "AAPL"
        :param resolution: 時間週期 - "MINUTE", "MINUTE_5", "MINUTE_15", "MINUTE_30", "HOUR", "HOUR_4", "DAY", "WEEK"
        :param start_date: 開始日期，格式 "YYYY-MM-DD" 或 "YYYY-MM-DDTHH:MM:SS"
        :param end_date: 結束日期，格式 "YYYY-MM-DD" 或 "YYYY-MM-DDTHH:MM:SS"
        :param max_results: 最大返回記錄數
        :return: 包含 OHLCV 數據的 DataFrame
        """
        if not self.cst or not self.x_security_token:
            self._login()

        # 格式化日期時間
        if "T" not in start_date:
            start_date = f"{start_date}T00:00:00"
        if "T" not in end_date:
            end_date = f"{end_date}T23:59:59"

        url = f"{self.base_url}/prices/{symbol}"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}
        params = {"resolution": resolution, "from": start_date, "to": end_date, "max": max_results}

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                response.json()
                prices = data.get("prices", [])

                if not prices:
                    logger.warning(f"沒有找到 {symbol} 在指定時間範圍內的數據")
                    return pd.DataFrame()

                # 轉換為DataFrame
                df_data = []
                for price in prices:
                    df_data.append(
                        {
                            "Date": pd.to_datetime(price["snapshotTime"]),
                            "Open": price["openPrice"]["ask"],
                            "High": price["highPrice"]["ask"],
                            "Low": price["lowPrice"]["ask"],
                            "Close": price["closePrice"]["ask"],
                            "Volume": 0,  # Capital.com API不提供成交量數據
                        }
                    )

                df = pd.DataFrame(df_data)
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)

                logger.info(f"成功獲取 {symbol} 從 {start_date} 到 {end_date} 的 {len(df)} 筆數據")
                return df

            else:
                logger.error(f"獲取歷史數據失敗: {response.status_code} - {response.text}")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            logger.error(f"獲取 {symbol} 歷史數據時出錯: {e}")
            return pd.DataFrame()

    def get_available_symbols(self) -> list:
        """獲取所有可用的交易品種"""
        if not self.cst or not self.x_security_token:
            self._login()

        url = f"{self.base_url}/markets"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}

        try:
            response = self.session.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                response.json()
                markets = data.get("markets", [])
                [market["epic"] for market in markets]
                logger.info(f"獲取到 {len(symbols)} 個可用交易品種")
                return symbols
            else:
                logger.error(f"獲取市場列表失敗: {response.status_code} - {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"獲取市場列表時出錯: {e}")
            return []

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """獲取特定市場的詳細信息"""
        if not self.cst or not self.x_security_token:
            self._login()

        url = f"{self.base_url}/markets/{symbol}"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}

        try:
            response = self.session.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"獲取市場信息失敗: {response.status_code} - {response.text}")
                return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"獲取市場信息時出錯: {e}")
            return {}

    def close(self):
        """關閉session"""
        if self.session:
            self.session.close()
