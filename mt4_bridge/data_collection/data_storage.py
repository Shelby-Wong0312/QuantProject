# -*- coding: utf-8 -*-
"""
高效數據存儲管理系統
支援即時查詢、數據壓縮和清理機制
"""

import asyncio
import logging
import pandas as pd
import sqlite3
import threading
import queue
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import time

from .tick_collector import TickData
from .ohlc_aggregator import OHLCBar, TimeFrame

logger = logging.getLogger(__name__)


class DataStorage:
    """高效數據存儲管理系統"""

    def __init__(
        self,
        storage_path: str = "./data/mt4_storage",
        enable_sqlite: bool = True,
        enable_parquet: bool = True,
        enable_csv: bool = False,
        enable_compression: bool = True,
        max_memory_cache: int = 100000,  # 內存中最大緩存記錄數
        auto_cleanup_days: int = 30,  # 自動清理超過N天的數據
        batch_size: int = 1000,  # 批量寫入大小
        flush_interval: int = 300,
    ):  # 刷新到磁盤的間隔(秒)
        """
        初始化數據存儲系統

        Args:
            storage_path: 存儲路徑
            enable_sqlite: 啟用SQLite存儲
            enable_parquet: 啟用Parquet存儲
            enable_csv: 啟用CSV存儲
            enable_compression: 啟用數據壓縮
            max_memory_cache: 最大內存緩存
            auto_cleanup_days: 自動清理天數
            batch_size: 批量處理大小
            flush_interval: 刷新間隔
        """
        self.storage_path = Path(storage_path)
        self.enable_sqlite = enable_sqlite
        self.enable_parquet = enable_parquet
        self.enable_csv = enable_csv
        self.enable_compression = enable_compression
        self.max_memory_cache = max_memory_cache
        self.auto_cleanup_days = auto_cleanup_days
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # 創建存儲目錄
        self._create_directories()

        # 內存緩存
        self.tick_cache: Dict[str, List[TickData]] = {}
        self.ohlc_cache: Dict[str, Dict[str, List[OHLCBar]]] = {}

        # 待寫入隊列
        self.write_queue = queue.Queue()
        self.write_thread = None

        # 控制標誌
        self._running = False
        self._last_flush = time.time()

        # 線程池
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="DataStorage"
        )

        # 線程鎖
        self._lock = threading.RLock()

        # 初始化數據庫
        if self.enable_sqlite:
            self._init_sqlite_db()

        # 統計信息
        self.stats = {
            "tick_records_stored": 0,
            "ohlc_records_stored": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup": None,
            "storage_size_mb": 0,
        }

        logger.info(f"數據存儲系統已初始化，路徑: {self.storage_path}")

    def _create_directories(self):
        """創建存儲目錄"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        if self.enable_sqlite:
            (self.storage_path / "sqlite").mkdir(exist_ok=True)
        if self.enable_parquet:
            (self.storage_path / "parquet").mkdir(exist_ok=True)
            (self.storage_path / "parquet" / "ticks").mkdir(exist_ok=True)
            (self.storage_path / "parquet" / "ohlc").mkdir(exist_ok=True)
        if self.enable_csv:
            (self.storage_path / "csv").mkdir(exist_ok=True)
            (self.storage_path / "csv" / "ticks").mkdir(exist_ok=True)
            (self.storage_path / "csv" / "ohlc").mkdir(exist_ok=True)

        # 壓縮數據目錄
        if self.enable_compression:
            (self.storage_path / "compressed").mkdir(exist_ok=True)

    def _init_sqlite_db(self):
        """初始化SQLite數據庫"""
        try:
            db_path = self.storage_path / "sqlite" / "mt4_data.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                # 創建Tick數據表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tick_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        bid REAL NOT NULL,
                        ask REAL NOT NULL,
                        last_price REAL NOT NULL,
                        volume INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # 創建OHLC數據表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ohlc_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume INTEGER DEFAULT 0,
                        tick_count INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # 創建索引
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data(symbol, timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_tf_time ON ohlc_data(symbol, timeframe, timestamp)"
                )

                conn.commit()
                logger.info("SQLite數據庫已初始化")

        except Exception as e:
            logger.error(f"初始化SQLite數據庫失敗: {e}")

    def start(self):
        """啟動數據存儲服務"""
        if self._running:
            logger.warning("數據存儲服務已在運行")
            return

        self._running = True

        # 啟動寫入線程
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()

        logger.info("數據存儲服務已啟動")

    def stop(self):
        """停止數據存儲服務"""
        if not self._running:
            return

        logger.info("正在停止數據存儲服務...")
        self._running = False

        # 刷新所有緩存數據
        self.flush_cache()

        # 等待寫入隊列完成
        self.write_queue.join()

        # 等待寫入線程結束
        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=10)

        # 關閉線程池
        self.executor.shutdown(wait=True)

        logger.info("數據存儲服務已停止")

    def _write_worker(self):
        """寫入工作線程"""
        while self._running or not self.write_queue.empty():
            try:
                task = self.write_queue.get(timeout=1)
                task_type, data = task

                if task_type == "tick":
                    self._write_tick_data(data)
                elif task_type == "ohlc":
                    self._write_ohlc_data(data)
                elif task_type == "flush":
                    self._flush_cached_data()

                self.write_queue.task_done()

            except queue.Empty:
                # 定期刷新緩存
                current_time = time.time()
                if current_time - self._last_flush >= self.flush_interval:
                    self._flush_cached_data()
                    self._last_flush = current_time

            except Exception as e:
                logger.error(f"寫入工作線程出錯: {e}")

    # === Tick 數據存儲 ===

    def store_tick(self, tick: TickData):
        """存儲Tick數據"""
        try:
            with self._lock:
                symbol = tick.symbol
                if symbol not in self.tick_cache:
                    self.tick_cache[symbol] = []

                self.tick_cache[symbol].append(tick)

                # 檢查是否需要批量寫入
                if len(self.tick_cache[symbol]) >= self.batch_size:
                    batch_data = self.tick_cache[symbol].copy()
                    self.tick_cache[symbol].clear()
                    self.write_queue.put(("tick", (symbol, batch_data)))

                # 檢查內存使用
                total_cached = sum(len(cache) for cache in self.tick_cache.values())
                if total_cached > self.max_memory_cache:
                    self.flush_cache()

        except Exception as e:
            logger.error(f"存儲Tick數據時出錯: {e}")

    def store_tick_batch(self, symbol: str, ticks: List[TickData]):
        """批量存儲Tick數據"""
        try:
            with self._lock:
                if symbol not in self.tick_cache:
                    self.tick_cache[symbol] = []

                self.tick_cache[symbol].extend(ticks)

                # 檢查是否需要寫入
                if len(self.tick_cache[symbol]) >= self.batch_size:
                    batch_data = self.tick_cache[symbol].copy()
                    self.tick_cache[symbol].clear()
                    self.write_queue.put(("tick", (symbol, batch_data)))

        except Exception as e:
            logger.error(f"批量存儲Tick數據時出錯: {e}")

    def _write_tick_data(self, data: Tuple[str, List[TickData]]):
        """寫入Tick數據到存儲"""
        symbol, ticks = data

        try:
            if not ticks:
                return

            # SQLite存儲
            if self.enable_sqlite:
                self._write_ticks_to_sqlite(symbol, ticks)

            # Parquet存儲
            if self.enable_parquet:
                self._write_ticks_to_parquet(symbol, ticks)

            # CSV存儲
            if self.enable_csv:
                self._write_ticks_to_csv(symbol, ticks)

            self.stats["tick_records_stored"] += len(ticks)
            logger.debug(f"已寫入 {len(ticks)} 筆 {symbol} Tick數據")

        except Exception as e:
            logger.error(f"寫入Tick數據時出錯: {e}")

    def _write_ticks_to_sqlite(self, symbol: str, ticks: List[TickData]):
        """寫入Tick數據到SQLite"""
        try:
            db_path = self.storage_path / "sqlite" / "mt4_data.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                data_to_insert = [
                    (
                        tick.symbol,
                        tick.timestamp.isoformat(),
                        tick.bid,
                        tick.ask,
                        tick.last,
                        tick.volume,
                    )
                    for tick in ticks
                ]

                cursor.executemany(
                    """
                    INSERT INTO tick_data (symbol, timestamp, bid, ask, last_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    data_to_insert,
                )

                conn.commit()

        except Exception as e:
            logger.error(f"寫入Tick數據到SQLite失敗: {e}")

    def _write_ticks_to_parquet(self, symbol: str, ticks: List[TickData]):
        """寫入Tick數據到Parquet"""
        try:
            df_data = [asdict(tick) for tick in ticks]
            df = pd.DataFrame(df_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            date_str = datetime.now().strftime("%Y%m%d")
            filename = (
                self.storage_path
                / "parquet"
                / "ticks"
                / f"{symbol}_ticks_{date_str}.parquet"
            )

            # 如果文件存在，追加數據
            if filename.exists():
                existing_df = pd.read_parquet(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
                # 去重
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.sort_values("timestamp")

            df.to_parquet(filename, index=False)

        except Exception as e:
            logger.error(f"寫入Tick數據到Parquet失敗: {e}")

    def _write_ticks_to_csv(self, symbol: str, ticks: List[TickData]):
        """寫入Tick數據到CSV"""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = (
                self.storage_path / "csv" / "ticks" / f"{symbol}_ticks_{date_str}.csv"
            )

            # 創建DataFrame
            df_data = [asdict(tick) for tick in ticks]
            df = pd.DataFrame(df_data)

            # 寫入或追加到CSV
            mode = "a" if filename.exists() else "w"
            header = not filename.exists()

            df.to_csv(filename, mode=mode, header=header, index=False)

        except Exception as e:
            logger.error(f"寫入Tick數據到CSV失敗: {e}")

    # === OHLC 數據存儲 ===

    def store_ohlc_bar(self, bar: OHLCBar):
        """存儲OHLC K線數據"""
        try:
            with self._lock:
                symbol = bar.symbol
                timeframe = bar.timeframe.value

                if symbol not in self.ohlc_cache:
                    self.ohlc_cache[symbol] = {}
                if timeframe not in self.ohlc_cache[symbol]:
                    self.ohlc_cache[symbol][timeframe] = []

                self.ohlc_cache[symbol][timeframe].append(bar)

                # 檢查是否需要批量寫入
                if len(self.ohlc_cache[symbol][timeframe]) >= self.batch_size:
                    batch_data = self.ohlc_cache[symbol][timeframe].copy()
                    self.ohlc_cache[symbol][timeframe].clear()
                    self.write_queue.put(("ohlc", (symbol, timeframe, batch_data)))

        except Exception as e:
            logger.error(f"存儲OHLC數據時出錯: {e}")

    def _write_ohlc_data(self, data: Tuple[str, str, List[OHLCBar]]):
        """寫入OHLC數據到存儲"""
        symbol, timeframe, bars = data

        try:
            if not bars:
                return

            # SQLite存儲
            if self.enable_sqlite:
                self._write_ohlc_to_sqlite(symbol, timeframe, bars)

            # Parquet存儲
            if self.enable_parquet:
                self._write_ohlc_to_parquet(symbol, timeframe, bars)

            # CSV存儲
            if self.enable_csv:
                self._write_ohlc_to_csv(symbol, timeframe, bars)

            self.stats["ohlc_records_stored"] += len(bars)
            logger.debug(f"已寫入 {len(bars)} 筆 {symbol} {timeframe} OHLC數據")

        except Exception as e:
            logger.error(f"寫入OHLC數據時出錯: {e}")

    def _write_ohlc_to_sqlite(self, symbol: str, timeframe: str, bars: List[OHLCBar]):
        """寫入OHLC數據到SQLite"""
        try:
            db_path = self.storage_path / "sqlite" / "mt4_data.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                data_to_insert = [
                    (
                        bar.symbol,
                        bar.timeframe.value,
                        bar.timestamp.isoformat(),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        bar.tick_count,
                    )
                    for bar in bars
                ]

                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO ohlc_data 
                    (symbol, timeframe, timestamp, open_price, high_price, low_price, 
                     close_price, volume, tick_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    data_to_insert,
                )

                conn.commit()

        except Exception as e:
            logger.error(f"寫入OHLC數據到SQLite失敗: {e}")

    def _write_ohlc_to_parquet(self, symbol: str, timeframe: str, bars: List[OHLCBar]):
        """寫入OHLC數據到Parquet"""
        try:
            df_data = [asdict(bar) for bar in bars]
            df = pd.DataFrame(df_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timeframe"] = df["timeframe"].apply(
                lambda x: x.value if hasattr(x, "value") else x
            )

            date_str = datetime.now().strftime("%Y%m%d")
            filename = (
                self.storage_path
                / "parquet"
                / "ohlc"
                / f"{symbol}_{timeframe}_ohlc_{date_str}.parquet"
            )

            # 如果文件存在，追加數據
            if filename.exists():
                existing_df = pd.read_parquet(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
                # 去重
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.sort_values("timestamp")

            df.to_parquet(filename, index=False)

        except Exception as e:
            logger.error(f"寫入OHLC數據到Parquet失敗: {e}")

    def _write_ohlc_to_csv(self, symbol: str, timeframe: str, bars: List[OHLCBar]):
        """寫入OHLC數據到CSV"""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = (
                self.storage_path
                / "csv"
                / "ohlc"
                / f"{symbol}_{timeframe}_ohlc_{date_str}.csv"
            )

            # 創建DataFrame
            df_data = [asdict(bar) for bar in bars]
            df = pd.DataFrame(df_data)
            df["timeframe"] = df["timeframe"].apply(
                lambda x: x.value if hasattr(x, "value") else x
            )

            # 寫入或追加到CSV
            mode = "a" if filename.exists() else "w"
            header = not filename.exists()

            df.to_csv(filename, mode=mode, header=header, index=False)

        except Exception as e:
            logger.error(f"寫入OHLC數據到CSV失敗: {e}")

    # === 查詢功能 ===

    def query_tick_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[TickData]:
        """查詢Tick數據"""
        try:
            self.stats["total_queries"] += 1

            # 先檢查緩存
            cached_data = self._query_tick_from_cache(
                symbol, start_time, end_time, limit
            )
            if cached_data:
                self.stats["cache_hits"] += 1
                return cached_data

            # 從存儲查詢
            self.stats["cache_misses"] += 1

            if self.enable_sqlite:
                return self._query_tick_from_sqlite(symbol, start_time, end_time, limit)
            elif self.enable_parquet:
                return self._query_tick_from_parquet(
                    symbol, start_time, end_time, limit
                )
            else:
                return []

        except Exception as e:
            logger.error(f"查詢Tick數據時出錯: {e}")
            return []

    def _query_tick_from_cache(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[TickData]:
        """從緩存查詢Tick數據"""
        try:
            with self._lock:
                if symbol not in self.tick_cache:
                    return []

                ticks = self.tick_cache[symbol]

                # 時間過濾
                if start_time or end_time:
                    filtered_ticks = []
                    for tick in ticks:
                        if start_time and tick.timestamp < start_time:
                            continue
                        if end_time and tick.timestamp > end_time:
                            continue
                        filtered_ticks.append(tick)
                    ticks = filtered_ticks

                # 限制數量
                if limit:
                    ticks = ticks[-limit:]

                return ticks

        except Exception as e:
            logger.error(f"從緩存查詢Tick數據時出錯: {e}")
            return []

    def _query_tick_from_sqlite(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[TickData]:
        """從SQLite查詢Tick數據"""
        try:
            db_path = self.storage_path / "sqlite" / "mt4_data.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                query = "SELECT symbol, timestamp, bid, ask, last_price, volume FROM tick_data WHERE symbol = ?"
                params = [symbol]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                ticks = []
                for row in rows:
                    tick = TickData(
                        symbol=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        bid=row[2],
                        ask=row[3],
                        last=row[4],
                        volume=row[5],
                    )
                    ticks.append(tick)

                return ticks[::-1]  # 返回時間順序

        except Exception as e:
            logger.error(f"從SQLite查詢Tick數據時出錯: {e}")
            return []

    def query_ohlc_data(
        self,
        symbol: str,
        timeframe: Union[str, TimeFrame],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCBar]:
        """查詢OHLC數據"""
        try:
            self.stats["total_queries"] += 1

            if isinstance(timeframe, TimeFrame):
                timeframe = timeframe.value

            # 先檢查緩存
            cached_data = self._query_ohlc_from_cache(
                symbol, timeframe, start_time, end_time, limit
            )
            if cached_data:
                self.stats["cache_hits"] += 1
                return cached_data

            # 從存儲查詢
            self.stats["cache_misses"] += 1

            if self.enable_sqlite:
                return self._query_ohlc_from_sqlite(
                    symbol, timeframe, start_time, end_time, limit
                )
            elif self.enable_parquet:
                return self._query_ohlc_from_parquet(
                    symbol, timeframe, start_time, end_time, limit
                )
            else:
                return []

        except Exception as e:
            logger.error(f"查詢OHLC數據時出錯: {e}")
            return []

    def _query_ohlc_from_sqlite(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCBar]:
        """從SQLite查詢OHLC數據"""
        try:
            db_path = self.storage_path / "sqlite" / "mt4_data.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                query = """
                SELECT symbol, timeframe, timestamp, open_price, high_price, 
                       low_price, close_price, volume, tick_count 
                FROM ohlc_data WHERE symbol = ? AND timeframe = ?
                """
                params = [symbol, timeframe]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                bars = []
                for row in rows:
                    bar = OHLCBar(
                        symbol=row[0],
                        timeframe=TimeFrame(row[1]),
                        timestamp=datetime.fromisoformat(row[2]),
                        open=row[3],
                        high=row[4],
                        low=row[5],
                        close=row[6],
                        volume=row[7],
                        tick_count=row[8],
                    )
                    bars.append(bar)

                return bars[::-1]  # 返回時間順序

        except Exception as e:
            logger.error(f"從SQLite查詢OHLC數據時出錯: {e}")
            return []

    # === 緩存管理 ===

    def flush_cache(self):
        """刷新所有緩存到磁盤"""
        try:
            with self._lock:
                # 刷新Tick緩存
                for symbol, ticks in self.tick_cache.items():
                    if ticks:
                        self.write_queue.put(("tick", (symbol, ticks.copy())))
                        ticks.clear()

                # 刷新OHLC緩存
                for symbol, timeframes in self.ohlc_cache.items():
                    for timeframe, bars in timeframes.items():
                        if bars:
                            self.write_queue.put(
                                ("ohlc", (symbol, timeframe, bars.copy()))
                            )
                            bars.clear()

                logger.info("已刷新所有緩存數據到寫入隊列")

        except Exception as e:
            logger.error(f"刷新緩存時出錯: {e}")

    def _flush_cached_data(self):
        """內部刷新方法"""
        self.flush_cache()
        self._last_flush = time.time()

    # === 數據清理 ===

    def cleanup_old_data(self, days: int = None):
        """清理舊數據"""
        try:
            cleanup_days = days or self.auto_cleanup_days
            cutoff_date = datetime.now() - timedelta(days=cleanup_days)

            if self.enable_sqlite:
                self._cleanup_sqlite_data(cutoff_date)

            if self.enable_parquet:
                self._cleanup_parquet_data(cutoff_date)

            if self.enable_csv:
                self._cleanup_csv_data(cutoff_date)

            self.stats["last_cleanup"] = datetime.now().isoformat()
            logger.info(f"已清理 {cleanup_days} 天前的舊數據")

        except Exception as e:
            logger.error(f"清理舊數據時出錯: {e}")

    def _cleanup_sqlite_data(self, cutoff_date: datetime):
        """清理SQLite中的舊數據"""
        try:
            db_path = self.storage_path / "sqlite" / "mt4_data.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                # 清理舊Tick數據
                cursor.execute(
                    "DELETE FROM tick_data WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )
                tick_deleted = cursor.rowcount

                # 清理舊OHLC數據
                cursor.execute(
                    "DELETE FROM ohlc_data WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )
                ohlc_deleted = cursor.rowcount

                conn.commit()

                logger.info(
                    f"已從SQLite刪除 {tick_deleted} 筆Tick和 {ohlc_deleted} 筆OHLC舊記錄"
                )

        except Exception as e:
            logger.error(f"清理SQLite舊數據時出錯: {e}")

    # === 統計和監控 ===

    def get_storage_statistics(self) -> Dict[str, Any]:
        """獲取存儲統計信息"""
        try:
            # 計算存儲大小
            total_size = 0
            for path in self.storage_path.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size

            self.stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)

            # 緩存統計
            cache_stats = {
                "tick_cache_size": sum(
                    len(cache) for cache in self.tick_cache.values()
                ),
                "ohlc_cache_size": sum(
                    sum(len(bars) for bars in timeframes.values())
                    for timeframes in self.ohlc_cache.values()
                ),
                "write_queue_size": self.write_queue.qsize(),
            }

            return {**self.stats, **cache_stats, "running": self._running}

        except Exception as e:
            logger.error(f"獲取統計信息時出錯: {e}")
            return self.stats

    def __del__(self):
        """析構函數"""
        try:
            self.stop()
        except Exception as e:
            logger.error(f"數據存儲析構時出錯: {e}")


# === 使用示例 ===


async def example_usage():
    """數據存儲系統使用示例"""
    from datetime import timedelta

    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 創建存儲系統
    storage = DataStorage(
        storage_path="./example_storage",
        enable_sqlite=True,
        enable_parquet=True,
        enable_compression=True,
        batch_size=100,
        flush_interval=10,
    )

    try:
        # 啟動存儲系統
        storage.start()

        # 模擬存儲一些Tick數據
        print("存儲模擬Tick數據...")
        for i in range(1000):
            tick = TickData(
                symbol="EURUSD",
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
                bid=1.1000 + i * 0.0001,
                ask=1.1010 + i * 0.0001,
            )
            storage.store_tick(tick)

        # 刷新緩存
        storage.flush_cache()

        # 等待寫入完成
        await asyncio.sleep(2)

        # 查詢數據
        print("查詢最近100筆Tick數據...")
        recent_ticks = storage.query_tick_data("EURUSD", limit=100)
        print(f"查詢到 {len(recent_ticks)} 筆Tick數據")

        # 顯示統計信息
        stats = storage.get_storage_statistics()
        print("\n=== 存儲統計 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

    finally:
        storage.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
