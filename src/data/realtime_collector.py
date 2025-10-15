"""
Realtime Data Collection System
實時數據收集系統
Cloud DE - Task DE-402
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import deque
import pickle
import gzip

# Redis alternative for Windows (using in-memory cache)
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory cache")

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Tick數據結構"""

    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "timestamp": self.timestamp.isoformat(),
        }


class InMemoryCache:
    """內存緩存 (Redis替代方案)"""

    def __init__(self):
        self.data = {}
        self.expiry = {}
        self.lock = threading.Lock()

    def set(self, key: str, value: str, ex: int = None):
        """設置鍵值"""
        with self.lock:
            self.data[key] = value
            if ex:
                self.expiry[key] = time.time() + ex

    def get(self, key: str) -> Optional[str]:
        """獲取值"""
        with self.lock:
            # 檢查過期
            if key in self.expiry:
                if time.time() > self.expiry[key]:
                    del self.data[key]
                    del self.expiry[key]
                    return None

            return self.data.get(key)

    def lpush(self, key: str, value: str):
        """列表推入"""
        with self.lock:
            if key not in self.data:
                self.data[key] = deque(maxlen=10000)
            self.data[key].appendleft(value)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        """獲取列表範圍"""
        with self.lock:
            if key not in self.data:
                return []

            data_list = list(self.data[key])
            if end == -1:
                return data_list[start:]
            return data_list[start : end + 1]

    def keys(self, pattern: str) -> List[str]:
        """獲取匹配的鍵"""
        import fnmatch

        with self.lock:
            return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]

    def delete(self, *keys):
        """刪除鍵"""
        with self.lock:
            for key in keys:
                self.data.pop(key, None)
                self.expiry.pop(key, None)


class RealtimeDataCollector:
    """
    實時數據收集器
    支援多線程並行收集和數據緩存
    """

    def __init__(
        self,
        symbols: List[str],
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_workers: int = 10,
    ):
        """
        初始化數據收集器

        Args:
            symbols: 股票代碼列表
            redis_host: Redis主機
            redis_port: Redis端口
            max_workers: 最大工作線程數
        """
        self.symbols = symbols
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 初始化緩存
        if REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                # 測試連接
                self.cache.ping()
                logger.info("Connected to Redis cache")
            except:
                logger.warning("Redis connection failed, using in-memory cache")
                self.cache = InMemoryCache()
        else:
            self.cache = InMemoryCache()

        # 數據流
        self.data_streams = {}
        self.is_collecting = False

        # 統計信息
        self.stats = {"total_ticks": 0, "errors": 0, "last_update": None}

        # 分鐘數據聚合
        self.minute_bars = {}
        self.current_minute_data = {}

        logger.info(f"Data collector initialized for {len(symbols)} symbols")

    async def start_collection(self):
        """開始數據收集"""
        self.is_collecting = True
        logger.info("Starting data collection...")

        # 為每個符號創建收集任務
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.collect_symbol_data(symbol))
            tasks.append(task)

        # 啟動數據聚合任務
        aggregation_task = asyncio.create_task(self.aggregate_minute_bars())
        tasks.append(aggregation_task)

        # 啟動統計報告任務
        stats_task = asyncio.create_task(self.report_stats())
        tasks.append(stats_task)

        await asyncio.gather(*tasks)

    async def collect_symbol_data(self, symbol: str):
        """
        收集單個符號的數據

        Args:
            symbol: 股票代碼
        """
        logger.info(f"Starting collection for {symbol}")

        while self.is_collecting:
            try:
                # 模擬數據接收 (實際應該從WebSocket或API獲取)
                tick = await self.simulate_tick_data(symbol)

                # 驗證數據
                if self.validate_tick(tick):
                    # 清洗數據
                    cleaned_tick = self.clean_tick_data(tick)

                    # 存儲到緩存
                    await self.store_tick(cleaned_tick)

                    # 更新分鐘數據
                    self.update_minute_data(cleaned_tick)

                    # 更新統計
                    self.stats["total_ticks"] += 1
                    self.stats["last_update"] = datetime.now()
                else:
                    self.stats["errors"] += 1
                    logger.warning(f"Invalid tick data for {symbol}")

                # 控制頻率
                await asyncio.sleep(0.1)  # 10 ticks per second per symbol

            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1)

    async def simulate_tick_data(self, symbol: str) -> TickData:
        """
        模擬tick數據 (用於測試)

        Args:
            symbol: 股票代碼

        Returns:
            Tick數據
        """
        # 基礎價格
        base_prices = {"AAPL": 180, "GOOGL": 140, "MSFT": 380, "AMZN": 170, "TSLA": 250}

        base_price = base_prices.get(symbol, 100)

        # 添加隨機變動
        price_change = np.random.normal(0, 0.001) * base_price
        new_price = base_price + price_change

        # 生成買賣價
        spread = 0.01  # 0.01% spread
        bid = new_price * (1 - spread / 2)
        ask = new_price * (1 + spread / 2)

        # 隨機成交量
        volume = np.random.randint(100, 10000)

        return TickData(
            symbol=symbol,
            price=new_price,
            volume=volume,
            bid=bid,
            ask=ask,
            timestamp=datetime.now(),
        )

    def validate_tick(self, tick: TickData) -> bool:
        """
        驗證tick數據

        Args:
            tick: Tick數據

        Returns:
            是否有效
        """
        # 檢查必要字段
        if not tick.symbol or not tick.timestamp:
            return False

        # 價格合理性檢查
        if tick.price <= 0 or tick.price > 1000000:
            return False

        # 買賣價檢查
        if tick.bid <= 0 or tick.ask <= 0:
            return False

        if tick.bid >= tick.ask:
            return False

        # 成交量檢查
        if tick.volume < 0:
            return False

        # 時間戳檢查
        time_diff = abs((datetime.now() - tick.timestamp).total_seconds())
        if time_diff > 60:  # 超過1分鐘的數據視為過期
            return False

        return True

    def clean_tick_data(self, tick: TickData) -> TickData:
        """
        清洗tick數據

        Args:
            tick: 原始tick數據

        Returns:
            清洗後的tick數據
        """
        # 四捨五入價格到合理精度
        tick.price = round(tick.price, 4)
        tick.bid = round(tick.bid, 4)
        tick.ask = round(tick.ask, 4)

        # 確保成交量為整數
        tick.volume = int(tick.volume)

        return tick

    async def store_tick(self, tick: TickData):
        """
        存儲tick數據到緩存

        Args:
            tick: Tick數據
        """
        try:
            # 生成鍵
            tick_key = f"tick:{tick.symbol}"
            timestamp_key = f"tick:{tick.symbol}:{tick.timestamp.timestamp()}"

            # 存儲到列表
            tick_json = json.dumps(tick.to_dict())
            self.cache.lpush(tick_key, tick_json)

            # 存儲帶時間戳的數據
            self.cache.set(timestamp_key, tick_json, ex=3600)  # 1小時過期

        except Exception as e:
            logger.error(f"Failed to store tick: {e}")

    def update_minute_data(self, tick: TickData):
        """
        更新分鐘數據

        Args:
            tick: Tick數據
        """
        current_minute = tick.timestamp.replace(second=0, microsecond=0)
        key = f"{tick.symbol}:{current_minute}"

        if key not in self.current_minute_data:
            self.current_minute_data[key] = {
                "symbol": tick.symbol,
                "timestamp": current_minute,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.volume,
                "tick_count": 1,
            }
        else:
            bar = self.current_minute_data[key]
            bar["high"] = max(bar["high"], tick.price)
            bar["low"] = min(bar["low"], tick.price)
            bar["close"] = tick.price
            bar["volume"] += tick.volume
            bar["tick_count"] += 1

    async def aggregate_minute_bars(self):
        """聚合分鐘K線"""
        while self.is_collecting:
            await asyncio.sleep(60)  # 每分鐘執行一次

            current_time = datetime.now()
            cutoff_time = current_time.replace(second=0, microsecond=0)

            # 處理完成的分鐘數據
            completed_bars = []
            for key in list(self.current_minute_data.keys()):
                bar = self.current_minute_data[key]
                if bar["timestamp"] < cutoff_time:
                    completed_bars.append(bar)
                    del self.current_minute_data[key]

            # 存儲完成的K線
            for bar in completed_bars:
                await self.store_minute_bar(bar)

            if completed_bars:
                logger.info(f"Aggregated {len(completed_bars)} minute bars")

    async def store_minute_bar(self, bar: Dict):
        """
        存儲分鐘K線

        Args:
            bar: K線數據
        """
        try:
            key = f"bar:1m:{bar['symbol']}"
            bar_json = json.dumps(bar, default=str)
            self.cache.lpush(key, bar_json)

        except Exception as e:
            logger.error(f"Failed to store minute bar: {e}")

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """
        獲取最近的tick數據

        Args:
            symbol: 股票代碼
            count: 數量

        Returns:
            Tick數據列表
        """
        try:
            key = f"tick:{symbol}"
            tick_strings = self.cache.lrange(key, 0, count - 1)

            ticks = []
            for tick_str in tick_strings:
                if isinstance(tick_str, str):
                    ticks.append(json.loads(tick_str))

            return ticks

        except Exception as e:
            logger.error(f"Failed to get recent ticks: {e}")
            return []

    def get_minute_bars(self, symbol: str, count: int = 60) -> pd.DataFrame:
        """
        獲取分鐘K線數據

        Args:
            symbol: 股票代碼
            count: K線數量

        Returns:
            K線DataFrame
        """
        try:
            key = f"bar:1m:{symbol}"
            bar_strings = self.cache.lrange(key, 0, count - 1)

            bars = []
            for bar_str in bar_strings:
                if isinstance(bar_str, str):
                    bars.append(json.loads(bar_str))

            if bars:
                df = pd.DataFrame(bars)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get minute bars: {e}")
            return pd.DataFrame()

    async def report_stats(self):
        """報告統計信息"""
        while self.is_collecting:
            await asyncio.sleep(30)  # 每30秒報告一次

            logger.info(
                f"Collection Stats - "
                f"Total ticks: {self.stats['total_ticks']}, "
                f"Errors: {self.stats['errors']}, "
                f"Last update: {self.stats['last_update']}"
            )

    async def backup_data(self, filepath: str):
        """
        備份數據到文件

        Args:
            filepath: 備份文件路徑
        """
        try:
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.symbols,
                "stats": self.stats,
                "recent_data": {},
            }

            # 收集每個符號的最近數據
            for symbol in self.symbols:
                backup_data["recent_data"][symbol] = {
                    "ticks": self.get_recent_ticks(symbol, 1000),
                    "bars": self.get_minute_bars(symbol, 60).to_dict(),
                }

            # 壓縮保存
            with gzip.open(filepath, "wb") as f:
                pickle.dump(backup_data, f)

            logger.info(f"Data backed up to {filepath}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")

    def stop_collection(self):
        """停止數據收集"""
        self.is_collecting = False
        self.executor.shutdown(wait=True)
        logger.info("Data collection stopped")


class DataValidator:
    """
    數據驗證器
    確保數據質量和完整性
    """

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        驗證OHLCV數據

        Args:
            df: OHLCV DataFrame

        Returns:
            驗證後的DataFrame
        """
        # 檢查必要列
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        # 移除無效行
        df = df[df["open"] > 0]
        df = df[df["high"] >= df["low"]]
        df = df[df["high"] >= df[["open", "close"]].max(axis=1)]
        df = df[df["low"] <= df[["open", "close"]].min(axis=1)]
        df = df[df["volume"] >= 0]

        # 填充缺失值
        df["volume"] = df["volume"].fillna(0)

        # 前向填充價格
        price_columns = ["open", "high", "low", "close"]
        df[price_columns] = df[price_columns].fillna(method="ffill")

        return df

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        檢查數據質量

        Args:
            df: 數據DataFrame

        Returns:
            質量報告
        """
        report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "date_gaps": [],
        }

        # 檢查時間間隔
        if not df.empty and df.index.name == "timestamp":
            time_diff = df.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=1)

            gaps = time_diff[time_diff > expected_diff * 1.5]
            if not gaps.empty:
                report["date_gaps"] = [
                    {
                        "start": str(df.index[i - 1]),
                        "end": str(df.index[i]),
                        "gap_minutes": gaps.iloc[i].total_seconds() / 60,
                    }
                    for i in range(len(gaps))
                ]

        return report


if __name__ == "__main__":
    # 測試數據收集器
    async def test_collector():
        symbols = ["AAPL", "GOOGL", "MSFT"]

        collector = RealtimeDataCollector(symbols)

        # 啟動收集
        collection_task = asyncio.create_task(collector.start_collection())

        # 運行10秒
        await asyncio.sleep(10)

        # 獲取數據
        for symbol in symbols:
            ticks = collector.get_recent_ticks(symbol, 10)
            print(f"\n{symbol} - Recent ticks: {len(ticks)}")

            if ticks:
                print(f"  Latest: {ticks[0]}")

        # 停止收集
        collector.stop_collection()

        # 備份數據
        await collector.backup_data("data/test_backup.pkl.gz")

        print("\nTest completed!")

    print("Testing Realtime Data Collector...")
    asyncio.run(test_collector())
