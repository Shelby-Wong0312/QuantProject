# quant_project/data/data_manager.py
# 統一數據管理器

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .capital_history_loader import CapitalHistoryLoader
from .data_cache import DataCache

logger = logging.getLogger(__name__)


class DataManager:
    """
    統一管理數據獲取，整合API調用和緩存機制
    """

    def __init__(self, use_cache: bool = True):
        self.capital_loader = CapitalHistoryLoader()
        self.cache = DataCache() if use_cache else None
        self.use_cache = use_cache

    def get_historical_data(
        self,
        symbol: str,
        resolution: str = "DAY",
        start_date: str = None,
        end_date: str = None,
        lookback_days: int = None,
    ) -> pd.DataFrame:
        """
        獲取歷史數據，優先從緩存讀取

        :param symbol: 股票代碼
        :param resolution: 時間週期
        :param start_date: 開始日期 (YYYY-MM-DD)
        :param end_date: 結束日期 (YYYY-MM-DD)
        :param lookback_days: 向前查看天數（如果沒有指定start_date）
        :return: DataFrame
        """
        # 處理日期參數
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if not start_date:
            if lookback_days:
                start_dt = datetime.now() - timedelta(days=lookback_days)
                start_date = start_dt.strftime("%Y-%m-%d")
            else:
                # 默認獲取一年數據
                start_dt = datetime.now() - timedelta(days=365)
                start_date = start_dt.strftime("%Y-%m-%d")

        # 嘗試從緩存獲取
        if self.use_cache and self.cache:
            cached_data = self.cache.get(symbol, resolution, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # 從API獲取數據
        logger.info(f"從Capital.com API獲取 {symbol} 數據...")
        self.capital_loader.get_bars(symbol, resolution, start_date, end_date)

        # 保存到緩存
        if self.use_cache and self.cache and not data.empty:
            self.cache.set(symbol, resolution, start_date, end_date, data)

        return data

    def get_multiple_symbols_data(
        self,
        symbols: List[str],
        resolution: str = "DAY",
        start_date: str = None,
        end_date: str = None,
        lookback_days: int = None,
        max_workers: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        並行獲取多個股票的歷史數據

        :param symbols: 股票代碼列表
        :param resolution: 時間週期
        :param start_date: 開始日期
        :param end_date: 結束日期
        :param lookback_days: 向前查看天數
        :param max_workers: 最大並行數
        :return: {symbol: DataFrame} 字典
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_symbol = {
                executor.submit(
                    self.get_historical_data,
                    symbol,
                    resolution,
                    start_date,
                    end_date,
                    lookback_days,
                ): symbol
                for symbol in symbols
            }

            # 收集結果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    future.result()
                    results[symbol] = data
                    logger.info(f"成功獲取 {symbol} 的數據 ({len(data)} 筆)")
                except Exception as e:
                    logger.error(f"獲取 {symbol} 數據失敗: {e}")
                    results[symbol] = pd.DataFrame()

                # 避免過快調用API
                time.sleep(0.1)

        return results

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        獲取最新價格

        :param symbol: 股票代碼
        :return: 最新價格或None
        """
        market_info = self.capital_loader.get_market_info(symbol)
        if market_info and "snapshot" in market_info:
            snapshot = market_info["snapshot"]
            if snapshot.get("offer") and snapshot.get("bid"):
                return (snapshot["offer"] + snapshot["bid"]) / 2
        return None

    def get_batch_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        批量獲取最新價格

        :param symbols: 股票代碼列表
        :return: {symbol: price} 字典
        """
        prices = {}
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price:
                prices[symbol] = price
            time.sleep(0.1)  # 避免過快調用
        return prices

    def update_cache(self, symbol: str, resolution: str = "DAY", days_back: int = 30):
        """
        更新特定股票的緩存數據

        :param symbol: 股票代碼
        :param resolution: 時間週期
        :param days_back: 更新多少天前的數據
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # 強制從API獲取新數據
        self.capital_loader.get_bars(symbol, resolution, start_date, end_date)

        if self.use_cache and self.cache and not data.empty:
            self.cache.set(symbol, resolution, start_date, end_date, data)
            logger.info(f"已更新 {symbol} 的緩存數據")

    def get_available_symbols(self) -> List[str]:
        """獲取所有可用的交易品種"""
        return self.capital_loader.get_available_symbols()

    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計信息"""
        if self.cache:
            return self.cache.get_cache_stats()
        return {}

    def clear_cache(self):
        """清理過期緩存"""
        if self.cache:
            self.cache.clear_expired()

    def close(self):
        """關閉連接"""
        self.capital_loader.close()
