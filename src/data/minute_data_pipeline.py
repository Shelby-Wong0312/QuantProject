"""
Minute-Level Data Pipeline
分鐘級數據處理管道 - 支援日內交易策略
Cloud DE - Task DT-003
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
import os
import asyncio
import aiohttp
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


class MinuteDataPipeline:
    """
    分鐘級數據處理管道

    功能：
    - 多數據源支援
    - 高效存儲和讀取
    - 數據清洗和驗證
    - 實時數據流模擬
    """

    def __init__(self, data_dir: str = "data/minute", cache_size: int = 128):
        """
        初始化數據管道

        Args:
            data_dir: 數據存儲目錄
            cache_size: LRU 緩存大小
        """
        self.data_dir = Path(data_dir)
        self.cache_size = cache_size

        # 創建目錄結構
        self._setup_directories()

        # 元數據
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # 線程池
        self.executor = ThreadPoolExecutor(max_workers=10)

        logger.info(f"MinuteDataPipeline initialized with data_dir: {self.data_dir}")

    def _setup_directories(self):
        """設置目錄結構"""
        intervals = ["1min", "5min", "15min", "30min", "60min"]
        for interval in intervals:
            (self.data_dir / interval).mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """載入元數據"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"symbols": {}, "last_update": None, "total_records": 0}

    def _save_metadata(self):
        """保存元數據"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def download_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "5min",
        source: str = "yfinance",
    ) -> Dict[str, pd.DataFrame]:
        """
        下載分鐘數據

        Args:
            symbols: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期
            interval: 時間間隔
            source: 數據源

        Returns:
            數據字典 {symbol: DataFrame}
        """
        logger.info(f"Downloading {len(symbols)} symbols from {start_date} to {end_date}")

        # 轉換日期
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        results = {}

        if source == "yfinance":
            # 使用線程池並行下載
            futures = []
            for symbol in symbols:
                future = self.executor.submit(
                    self._download_yfinance, symbol, start_date, end_date, interval
                )
                futures.append((symbol, future))

            # 收集結果
            for symbol, future in futures:
                try:
                    df = future.result(timeout=30)
                    if df is not None and not df.empty:
                        # 清洗數據
                        df = self.clean_data(df)
                        # 保存到文件
                        self._save_to_parquet(df, symbol, interval)
                        results[symbol] = df
                        logger.info(f"Downloaded {symbol}: {len(df)} records")
                except Exception as e:
                    logger.error(f"Failed to download {symbol}: {e}")

        # 更新元數據
        self._update_metadata(symbols, start_date, end_date, interval)

        return results

    def _download_yfinance(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """使用 yfinance 下載數據"""
        try:
            # yfinance 限制：1分鐘數據只能獲取最近7天
            if interval == "1min":
                max_days = 7
                if (end_date - start_date).days > max_days:
                    start_date = end_date - timedelta(days=max_days)
            elif interval == "5min":
                max_days = 60
                if (end_date - start_date).days > max_days:
                    start_date = end_date - timedelta(days=max_days)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return None

            # 重命名列
            df.columns = df.columns.str.lower()
            df.index.name = "timestamp"

            return df

        except Exception as e:
            logger.error(f"yfinance download error for {symbol}: {e}")
            return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗數據

        Args:
            df: 原始數據

        Returns:
            清洗後的數據
        """
        # 移除重複
        df = df[~df.index.duplicated(keep="first")]

        # 處理缺失值
        df = df.fillna(method="ffill").fillna(method="bfill")

        # 移除異常值（價格為0或負數）
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]

        # 確保 high >= low
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)

        # 排序索引
        df = df.sort_index()

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        驗證數據質量

        Args:
            df: 數據 DataFrame

        Returns:
            驗證報告
        """
        report = {
            "total_records": len(df),
            "date_range": (df.index.min(), df.index.max()),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.index.duplicated().sum(),
            "price_errors": 0,
            "volume_errors": 0,
            "time_gaps": [],
        }

        # 檢查價格錯誤
        price_errors = (df["high"] < df["low"]).sum()
        report["price_errors"] = int(price_errors)

        # 檢查成交量錯誤
        volume_errors = (df["volume"] < 0).sum()
        report["volume_errors"] = int(volume_errors)

        # 檢查時間間隔
        time_diff = df.index.to_series().diff()
        expected_freq = pd.Timedelta(minutes=5)  # 假設5分鐘
        gaps = time_diff[time_diff > expected_freq * 2]
        if len(gaps) > 0:
            report["time_gaps"] = [(str(idx), str(gap)) for idx, gap in gaps.items()]

        # 計算質量分數
        quality_score = 100.0
        quality_score -= price_errors / len(df) * 100
        quality_score -= volume_errors / len(df) * 100
        quality_score -= len(gaps) / len(df) * 100
        report["quality_score"] = max(0, quality_score)

        return report

    def resample_data(
        self, df: pd.DataFrame, source_interval: str, target_interval: str
    ) -> pd.DataFrame:
        """
        重採樣數據

        Args:
            df: 原始數據
            source_interval: 源間隔
            target_interval: 目標間隔

        Returns:
            重採樣後的數據
        """
        # 轉換為 pandas 頻率字符串
        freq_map = {"1min": "1T", "5min": "5T", "15min": "15T", "30min": "30T", "60min": "60T"}

        target_freq = freq_map.get(target_interval, "5T")

        # 重採樣
        resampled = (
            df.resample(target_freq)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

        return resampled

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技術特徵

        Args:
            df: OHLCV 數據

        Returns:
            添加特徵後的數據
        """
        # 收益率
        df["returns"] = df["close"].pct_change()

        # 對數收益率
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # 波動率（滾動標準差）
        df["volatility"] = df["returns"].rolling(window=20).std()

        # 成交量變化率
        df["volume_change"] = df["volume"].pct_change()

        # 價格範圍
        df["price_range"] = (df["high"] - df["low"]) / df["close"]

        # 成交量加權平均價
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        # 價格位置
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        return df

    def _save_to_parquet(self, df: pd.DataFrame, symbol: str, interval: str):
        """保存數據到 Parquet 文件"""
        # 生成文件名
        start_date = df.index.min().strftime("%Y%m%d")
        end_date = df.index.max().strftime("%Y%m%d")
        filename = f"{symbol}_{start_date}_{end_date}.parquet"
        filepath = self.data_dir / interval / filename

        # 保存
        df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {symbol} to {filepath}")

    @lru_cache(maxsize=128)
    def load_data(
        self,
        symbol: str,
        interval: str = "5min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        載入數據（帶緩存）

        Args:
            symbol: 股票代碼
            interval: 時間間隔
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            數據 DataFrame
        """
        # 查找符合的文件
        data_path = self.data_dir / interval
        pattern = f"{symbol}_*.parquet"

        files = list(data_path.glob(pattern))
        if not files:
            logger.warning(f"No data files found for {symbol} at {interval}")
            return None

        # 載入所有文件
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            dfs.append(df)

        # 合併
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # 過濾日期範圍
        if start_date:
            combined = combined[combined.index >= start_date]
        if end_date:
            combined = combined[combined.index <= end_date]

        return combined

    def _update_metadata(
        self, symbols: List[str], start_date: datetime, end_date: datetime, interval: str
    ):
        """更新元數據"""
        for symbol in symbols:
            if symbol not in self.metadata["symbols"]:
                self.metadata["symbols"][symbol] = {}

            self.metadata["symbols"][symbol][interval] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "last_update": datetime.now().isoformat(),
            }

        self.metadata["last_update"] = datetime.now().isoformat()
        self._save_metadata()

    def get_statistics(self) -> Dict:
        """獲取數據統計"""
        stats = {
            "total_symbols": len(self.metadata["symbols"]),
            "intervals": {},
            "total_files": 0,
            "total_size_mb": 0,
        }

        # 統計各間隔的文件
        for interval_dir in self.data_dir.iterdir():
            if interval_dir.is_dir() and interval_dir.name != "metadata.json":
                files = list(interval_dir.glob("*.parquet"))
                stats["intervals"][interval_dir.name] = len(files)
                stats["total_files"] += len(files)

                # 計算總大小
                for file in files:
                    stats["total_size_mb"] += file.stat().st_size / (1024 * 1024)

        return stats


class MinuteDataStreamer:
    """
    實時數據流模擬器

    模擬實時數據流，用於策略測試
    """

    def __init__(self, data: pd.DataFrame, speed: float = 1.0):
        """
        初始化流模擬器

        Args:
            data: 歷史數據
            speed: 播放速度倍數
        """
        self.data = data
        self.speed = speed
        self.current_idx = 0
        self.is_streaming = False
        self.callbacks = []

    def add_callback(self, callback):
        """添加回調函數"""
        self.callbacks.append(callback)

    async def start_stream(self):
        """開始數據流"""
        self.is_streaming = True
        self.current_idx = 0

        while self.is_streaming and self.current_idx < len(self.data):
            # 獲取當前數據
            current_data = self.data.iloc[self.current_idx]

            # 觸發回調
            for callback in self.callbacks:
                await callback(current_data)

            # 等待（模擬實時）
            await asyncio.sleep(1.0 / self.speed)

            self.current_idx += 1

        self.is_streaming = False

    def stop_stream(self):
        """停止數據流"""
        self.is_streaming = False

    def reset(self):
        """重置流"""
        self.current_idx = 0
        self.is_streaming = False


# 便捷 API
class MinuteData:
    """便捷的數據訪問 API"""

    _pipeline = None

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            cls._pipeline = MinuteDataPipeline()
        return cls._pipeline

    @classmethod
    def get(
        cls, symbols: Union[str, List[str]], start: str, end: str, interval: str = "5min"
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        獲取分鐘數據

        Args:
            symbols: 股票代碼
            start: 開始日期
            end: 結束日期
            interval: 時間間隔

        Returns:
            數據 DataFrame 或字典
        """
        pipeline = cls._get_pipeline()

        # 單個股票
        if isinstance(symbols, str):
            # 嘗試載入
            data = pipeline.load_data(symbols, interval, start, end)

            # 如果沒有，下載
            if data is None or data.empty:
                results = pipeline.download_data([symbols], start, end, interval)
                data = results.get(symbols)

            return data

        # 多個股票
        results = {}
        symbols_to_download = []

        for symbol in symbols:
            data = pipeline.load_data(symbol, interval, start, end)
            if data is not None and not data.empty:
                results[symbol] = data
            else:
                symbols_to_download.append(symbol)

        # 下載缺失的
        if symbols_to_download:
            downloaded = pipeline.download_data(symbols_to_download, start, end, interval)
            results.update(downloaded)

        return results

    @classmethod
    def stream(cls, symbols: List[str], callback, interval: str = "5min"):
        """
        創建實時數據流

        Args:
            symbols: 股票代碼
            callback: 回調函數
            interval: 時間間隔

        Returns:
            數據流對象
        """
        # 這裡簡化實現，實際應連接真實數據源
        pipeline = cls._get_pipeline()

        # 獲取最近數據
        end = datetime.now()
        start = end - timedelta(days=1)
        data = pipeline.download_data(symbols, start, end, interval)

        if data and symbols[0] in data:
            streamer = MinuteDataStreamer(data[symbols[0]])
            streamer.add_callback(callback)
            return streamer

        return None


if __name__ == "__main__":
    print("Minute Data Pipeline - Cloud DE Task DT-003")
    print("=" * 50)

    # 測試數據下載
    print("\nTesting data download...")
    pipeline = MinuteDataPipeline()

    # 下載測試數據
    symbols = ["AAPL", "GOOGL", "MSFT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    results = pipeline.download_data(
        symbols=symbols, start_date=start_date, end_date=end_date, interval="5min"
    )

    print(f"\nDownloaded {len(results)} symbols")
    for symbol, df in results.items():
        print(f"  {symbol}: {len(df)} records")

        # 驗證數據
        report = pipeline.validate_data(df)
        print(f"    Quality score: {report['quality_score']:.2f}%")

    # 測試統計
    stats = pipeline.get_statistics()
    print(f"\nData Statistics:")
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")

    print("\n✓ Minute data pipeline ready!")
