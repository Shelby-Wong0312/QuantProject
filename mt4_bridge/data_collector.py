# -*- coding: utf-8 -*-
"""
MT4數據收集器模組
負責接收MT4的tick數據、K線數據聚合、技術指標計算介面
支援多種數據格式和存儲方案
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import deque, defaultdict
from threading import Lock, Thread
from dataclasses import dataclass, asdict
from enum import Enum
import os

from .connector import MT4Connector, get_default_connector

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """時間框架枚舉"""

    M1 = "M1"  # 1分鐘
    M5 = "M5"  # 5分鐘
    M15 = "M15"  # 15分鐘
    M30 = "M30"  # 30分鐘
    H1 = "H1"  # 1小時
    H4 = "H4"  # 4小時
    D1 = "D1"  # 日線
    W1 = "W1"  # 週線
    MN1 = "MN1"  # 月線


@dataclass
class TickData:
    """Tick數據結構"""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    spread: float
    volume: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "volume": self.volume,
        }


@dataclass
class OHLCData:
    """OHLC K線數據結構"""

    symbol: str
    timestamp: datetime
    timeframe: TimeFrame
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
        }


class DataStorage:
    """數據存儲介面"""

    def __init__(self, db_path: str = None):
        """
        初始化數據存儲

        Args:
            db_path: SQLite數據庫路徑，默認為項目根目錄下的mt4_data.db
        """
        if db_path is None:
            # 使用項目根目錄下的test_storage/sqlite目錄
            project_root = os.path.dirname(os.path.dirname(__file__))
            storage_dir = os.path.join(project_root, "test_storage", "sqlite")
            os.makedirs(storage_dir, exist_ok=True)
            db_path = os.path.join(storage_dir, "mt4_data.db")

        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化數據庫表"""
        with sqlite3.connect(self.db_path) as conn:
            # Tick數據表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    spread REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # OHLC數據表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    tick_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """
            )

            # 創建索引
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tick_symbol_timestamp ON tick_data(symbol, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timestamp ON ohlc_data(symbol, timestamp, timeframe)"
            )

    def save_tick(self, tick: TickData):
        """保存Tick數據"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO tick_data (symbol, timestamp, bid, ask, spread, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (tick.symbol, tick.timestamp, tick.bid, tick.ask, tick.spread, tick.volume),
                )
        except Exception as e:
            logger.error(f"保存Tick數據失敗: {e}")

    def save_ohlc(self, ohlc: OHLCData):
        """保存OHLC數據"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ohlc_data 
                    (symbol, timestamp, timeframe, open, high, low, close, volume, tick_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ohlc.symbol,
                        ohlc.timestamp,
                        ohlc.timeframe.value,
                        ohlc.open,
                        ohlc.high,
                        ohlc.low,
                        ohlc.close,
                        ohlc.volume,
                        ohlc.tick_count,
                    ),
                )
        except Exception as e:
            logger.error(f"保存OHLC數據失敗: {e}")

    def get_tick_data(
        self, symbol: str, start_time: datetime = None, end_time: datetime = None, limit: int = None
    ) -> List[TickData]:
        """獲取Tick數據"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT symbol, timestamp, bid, ask, spread, volume FROM tick_data WHERE symbol = ?"
                params = [symbol]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += f" LIMIT {limit}"

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                return [
                    TickData(
                        symbol=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        bid=row[2],
                        ask=row[3],
                        spread=row[4],
                        volume=row[5] or 0,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"獲取Tick數據失敗: {e}")
            return []

    def get_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = None,
    ) -> List[OHLCData]:
        """獲取OHLC數據"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT symbol, timestamp, timeframe, open, high, low, close, volume, tick_count 
                    FROM ohlc_data WHERE symbol = ? AND timeframe = ?
                """
                params = [symbol, timeframe.value]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += f" LIMIT {limit}"

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                return [
                    OHLCData(
                        symbol=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        timeframe=TimeFrame(row[2]),
                        open=row[3],
                        high=row[4],
                        low=row[5],
                        close=row[6],
                        volume=row[7],
                        tick_count=row[8] or 0,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"獲取OHLC數據失敗: {e}")
            return []


class OHLCAggregator:
    """OHLC聚合器 - 將Tick數據聚合為不同時間框架的K線"""

    def __init__(self):
        self.current_bars = {}  # {(symbol, timeframe): OHLCData}
        self.tick_counts = defaultdict(int)  # {(symbol, timeframe): count}
        self.lock = Lock()

    def add_tick(self, tick: TickData) -> List[OHLCData]:
        """
        添加Tick數據並返回完成的K線

        Args:
            tick: Tick數據

        Returns:
            List[OHLCData]: 完成的K線數據列表
        """
        completed_bars = []
        mid_price = (tick.bid + tick.ask) / 2.0

        with self.lock:
            # 為每個時間框架聚合數據
            for timeframe in TimeFrame:
                bar_time = self._get_bar_timestamp(tick.timestamp, timeframe)
                key = (tick.symbol, timeframe)

                # 檢查是否需要創建新的K線或更新現有K線
                if key not in self.current_bars:
                    # 創建新的K線
                    self.current_bars[key] = OHLCData(
                        symbol=tick.symbol,
                        timestamp=bar_time,
                        timeframe=timeframe,
                        open=mid_price,
                        high=mid_price,
                        low=mid_price,
                        close=mid_price,
                        volume=tick.volume,
                        tick_count=1,
                    )
                    self.tick_counts[key] = 1
                else:
                    current_bar = self.current_bars[key]

                    # 檢查是否需要開始新的K線
                    if bar_time > current_bar.timestamp:
                        # 當前K線已完成
                        completed_bars.append(current_bar)

                        # 創建新的K線
                        self.current_bars[key] = OHLCData(
                            symbol=tick.symbol,
                            timestamp=bar_time,
                            timeframe=timeframe,
                            open=mid_price,
                            high=mid_price,
                            low=mid_price,
                            close=mid_price,
                            volume=tick.volume,
                            tick_count=1,
                        )
                        self.tick_counts[key] = 1
                    else:
                        # 更新當前K線
                        current_bar.high = max(current_bar.high, mid_price)
                        current_bar.low = min(current_bar.low, mid_price)
                        current_bar.close = mid_price
                        current_bar.volume += tick.volume
                        current_bar.tick_count += 1
                        self.tick_counts[key] += 1

        return completed_bars

    def _get_bar_timestamp(self, timestamp: datetime, timeframe: TimeFrame) -> datetime:
        """獲取K線的標準時間戳"""
        if timeframe == TimeFrame.M1:
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == TimeFrame.M5:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.M15:
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.M30:
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.H1:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.H4:
            hour = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.D1:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.W1:
            # 週一開始
            days_since_monday = timestamp.weekday()
            start_of_week = timestamp - timedelta(days=days_since_monday)
            return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.MN1:
            # 月初開始
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp

    def get_current_bar(self, symbol: str, timeframe: TimeFrame) -> Optional[OHLCData]:
        """獲取當前未完成的K線"""
        with self.lock:
            return self.current_bars.get((symbol, timeframe))

    def force_close_bar(self, symbol: str, timeframe: TimeFrame) -> Optional[OHLCData]:
        """強制關閉當前K線"""
        with self.lock:
            key = (symbol, timeframe)
            if key in self.current_bars:
                bar = self.current_bars.pop(key)
                self.tick_counts.pop(key, 0)
                return bar
            return None


class TechnicalIndicators:
    """技術指標計算器"""

    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """簡單移動平均線"""
        if len(data) < period:
            return []

        result = []
        for i in range(period - 1, len(data)):
            avg = sum(data[i - period + 1 : i + 1]) / period
            result.append(avg)
        return result

    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """指數移動平均線"""
        if len(data) < period:
            return []

        alpha = 2.0 / (period + 1)
        result = []

        # 第一個值使用SMA
        sma_first = sum(data[:period]) / period
        result.append(sma_first)

        # 後續使用EMA公式
        for i in range(period, len(data)):
            ema = alpha * data[i] + (1 - alpha) * result[-1]
            result.append(ema)

        return result

    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """相對強弱指標"""
        if len(data) < period + 1:
            return []

        deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        avg_gains = TechnicalIndicators.sma(gains, period)
        avg_losses = TechnicalIndicators.sma(losses, period)

        rsi_values = []
        for i in range(len(avg_gains)):
            if avg_losses[i] == 0:
                rsi = 100
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        return rsi_values

    @staticmethod
    def bollinger_bands(
        data: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, List[float]]:
        """布林帶"""
        if len(data) < period:
            return {"upper": [], "middle": [], "lower": []}

        middle = TechnicalIndicators.sma(data, period)
        upper = []
        lower = []

        for i in range(period - 1, len(data)):
            window_data = data[i - period + 1 : i + 1]
            std = np.std(window_data)
            sma = middle[i - period + 1]
            upper.append(sma + std_dev * std)
            lower.append(sma - std_dev * std)

        return {"upper": upper, "middle": middle, "lower": lower}


class MT4DataCollector:
    """MT4數據收集器主類"""

    def __init__(self, connector: MT4Connector = None, storage: DataStorage = None):
        """
        初始化數據收集器

        Args:
            connector: MT4連接器，默認使用全局默認連接器
            storage: 數據存儲器，默認創建新的存儲器
        """
        self.connector = connector or get_default_connector()
        self.storage = storage or DataStorage()
        self.aggregator = OHLCAggregator()

        # 數據處理設置
        self.subscribed_symbols = set()
        self.tick_callbacks = []  # [(callback, symbols)]
        self.ohlc_callbacks = []  # [(callback, symbols, timeframes)]

        # 統計信息
        self.tick_count = 0
        self.ohlc_count = 0
        self.start_time = None

        # 線程控制
        self._running = False

    def start(self):
        """開始數據收集"""
        if not self.connector or not self.connector.is_connected():
            logger.error("MT4連接器未連接，無法開始數據收集")
            return False

        self._running = True
        self.start_time = datetime.now()

        # 訂閱數據流
        self.connector.subscribe_data("tick_data", self._on_tick_data)
        self.connector.subscribe_data("ohlc_data", self._on_ohlc_data)

        logger.info("MT4數據收集器已啟動")
        return True

    def stop(self):
        """停止數據收集"""
        self._running = False

        # 取消訂閱
        if self.connector:
            self.connector.unsubscribe_data("tick_data", self._on_tick_data)
            self.connector.unsubscribe_data("ohlc_data", self._on_ohlc_data)

        logger.info("MT4數據收集器已停止")

    def subscribe_symbol(self, symbol: str):
        """訂閱交易品種的數據"""
        self.subscribed_symbols.add(symbol)

        # 向MT4發送訂閱命令
        if self.connector:
            response = self.connector.send_command("SUBSCRIBE_SYMBOL", symbol=symbol)
            if response and response.get("success"):
                logger.info(f"已訂閱交易品種: {symbol}")
            else:
                logger.error(f"訂閱交易品種失敗: {symbol}")

    def unsubscribe_symbol(self, symbol: str):
        """取消訂閱交易品種"""
        self.subscribed_symbols.discard(symbol)

        # 向MT4發送取消訂閱命令
        if self.connector:
            response = self.connector.send_command("UNSUBSCRIBE_SYMBOL", symbol=symbol)
            if response and response.get("success"):
                logger.info(f"已取消訂閱交易品種: {symbol}")

    def add_tick_callback(self, callback: Callable[[TickData], None], symbols: List[str] = None):
        """
        添加Tick數據回調函數

        Args:
            callback: 回調函數，接收TickData參數
            symbols: 關注的交易品種列表，None表示所有品種
        """
        self.tick_callbacks.append((callback, symbols))

    def add_ohlc_callback(
        self,
        callback: Callable[[OHLCData], None],
        symbols: List[str] = None,
        timeframes: List[TimeFrame] = None,
    ):
        """
        添加OHLC數據回調函數

        Args:
            callback: 回調函數，接收OHLCData參數
            symbols: 關注的交易品種列表，None表示所有品種
            timeframes: 關注的時間框架列表，None表示所有時間框架
        """
        self.ohlc_callbacks.append((callback, symbols, timeframes))

    def _on_tick_data(self, data: Dict[str, Any]):
        """處理接收到的Tick數據"""
        try:
            # 解析Tick數據
            tick = TickData(
                symbol=data.get("symbol"),
                timestamp=datetime.fromisoformat(data.get("timestamp")),
                bid=float(data.get("bid")),
                ask=float(data.get("ask")),
                spread=float(data.get("spread", 0)),
                volume=int(data.get("volume", 0)),
            )

            # 只處理已訂閱的交易品種
            if tick.symbol not in self.subscribed_symbols:
                return

            # 保存到存儲
            self.storage.save_tick(tick)

            # 聚合為OHLC數據
            completed_bars = self.aggregator.add_tick(tick)
            for bar in completed_bars:
                self.storage.save_ohlc(bar)
                self._trigger_ohlc_callbacks(bar)

            # 觸發Tick回調
            self._trigger_tick_callbacks(tick)

            # 更新統計
            self.tick_count += 1

        except Exception as e:
            logger.error(f"處理Tick數據時發生錯誤: {e}")

    def _on_ohlc_data(self, data: Dict[str, Any]):
        """處理接收到的OHLC數據"""
        try:
            # 解析OHLC數據
            ohlc = OHLCData(
                symbol=data.get("symbol"),
                timestamp=datetime.fromisoformat(data.get("timestamp")),
                timeframe=TimeFrame(data.get("timeframe")),
                open=float(data.get("open")),
                high=float(data.get("high")),
                low=float(data.get("low")),
                close=float(data.get("close")),
                volume=int(data.get("volume", 0)),
                tick_count=int(data.get("tick_count", 0)),
            )

            # 只處理已訂閱的交易品種
            if ohlc.symbol not in self.subscribed_symbols:
                return

            # 保存到存儲
            self.storage.save_ohlc(ohlc)

            # 觸發回調
            self._trigger_ohlc_callbacks(ohlc)

            # 更新統計
            self.ohlc_count += 1

        except Exception as e:
            logger.error(f"處理OHLC數據時發生錯誤: {e}")

    def _trigger_tick_callbacks(self, tick: TickData):
        """觸發Tick回調函數"""
        for callback, symbols in self.tick_callbacks:
            try:
                if symbols is None or tick.symbol in symbols:
                    callback(tick)
            except Exception as e:
                logger.error(f"Tick回調函數執行錯誤: {e}")

    def _trigger_ohlc_callbacks(self, ohlc: OHLCData):
        """觸發OHLC回調函數"""
        for callback, symbols, timeframes in self.ohlc_callbacks:
            try:
                symbol_match = symbols is None or ohlc.symbol in symbols
                timeframe_match = timeframes is None or ohlc.timeframe in timeframes

                if symbol_match and timeframe_match:
                    callback(ohlc)
            except Exception as e:
                logger.error(f"OHLC回調函數執行錯誤: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """獲取數據收集統計信息"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "subscribed_symbols": list(self.subscribed_symbols),
            "tick_count": self.tick_count,
            "ohlc_count": self.ohlc_count,
            "ticks_per_second": self.tick_count / uptime if uptime > 0 else 0,
            "current_bars_count": len(self.aggregator.current_bars),
        }

    def get_latest_data(
        self, symbol: str, timeframe: TimeFrame, limit: int = 100
    ) -> List[OHLCData]:
        """獲取最新的歷史數據"""
        return self.storage.get_ohlc_data(symbol, timeframe, limit=limit)

    def calculate_indicators(
        self, symbol: str, timeframe: TimeFrame, indicator: str, **params
    ) -> List[float]:
        """
        計算技術指標

        Args:
            symbol: 交易品種
            timeframe: 時間框架
            indicator: 指標名稱 ("sma", "ema", "rsi", "bollinger_bands")
            **params: 指標參數

        Returns:
            List[float]: 指標值
        """
        # 獲取歷史數據
        ohlc_data = self.get_latest_data(symbol, timeframe, limit=params.get("limit", 1000))
        if not ohlc_data:
            return []

        # 提取收盤價
        close_prices = [bar.close for bar in reversed(ohlc_data)]  # 按時間正序

        # 計算指標
        if indicator.lower() == "sma":
            return TechnicalIndicators.sma(close_prices, params.get("period", 20))
        elif indicator.lower() == "ema":
            return TechnicalIndicators.ema(close_prices, params.get("period", 20))
        elif indicator.lower() == "rsi":
            return TechnicalIndicators.rsi(close_prices, params.get("period", 14))
        elif indicator.lower() == "bollinger_bands":
            return TechnicalIndicators.bollinger_bands(
                close_prices, params.get("period", 20), params.get("std_dev", 2.0)
            )
        else:
            logger.error(f"未知的指標: {indicator}")
            return []


# 便利函數
def create_data_collector(connector: MT4Connector = None) -> MT4DataCollector:
    """創建數據收集器實例"""
    return MT4DataCollector(connector)


def get_tick_data(symbol: str, hours: int = 1) -> List[TickData]:
    """獲取指定時間範圍的Tick數據"""
    storage = DataStorage()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    return storage.get_tick_data(symbol, start_time, end_time)


def get_ohlc_data(symbol: str, timeframe: TimeFrame, hours: int = 24) -> List[OHLCData]:
    """獲取指定時間範圍的OHLC數據"""
    storage = DataStorage()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    return storage.get_ohlc_data(symbol, timeframe, start_time, end_time)
