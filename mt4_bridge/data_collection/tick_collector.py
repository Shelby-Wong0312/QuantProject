# -*- coding: utf-8 -*-
"""
MT4 Tick 數據收集器
實現實時 Tick 級數據收集，支援多品種同時收集，具備實時緩存和持久化功能
"""

import asyncio
import logging
import json
import time
import pandas as pd
from datetime import datetime, timezone
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import threading
import queue
import csv
from concurrent.futures import ThreadPoolExecutor

from ..zeromq.python_side import MT4Bridge

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Tick 數據結構"""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float = None
    volume: int = 0

    def __post_init__(self):
        if self.last is None:
            self.last = (self.bid + self.ask) / 2

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TickData":
        """從字典創建 TickData"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class TickCollector:
    """MT4 Tick 數據收集器"""

    def __init__(
        self,
        symbols: List[str],
        mt4_bridge: Optional[MT4Bridge] = None,
        cache_size: int = 10000,
        storage_path: str = "./data/ticks",
        auto_save_interval: int = 300,  # 5分鐘自動保存
        enable_csv: bool = True,
        enable_parquet: bool = True,
    ):
        """
        初始化 Tick 收集器

        Args:
            symbols: 要收集的交易品種列表
            mt4_bridge: MT4橋接實例
            cache_size: 每個品種的緩存大小
            storage_path: 數據存儲路徑
            auto_save_interval: 自動保存間隔(秒)
            enable_csv: 是否啟用CSV存儲
            enable_parquet: 是否啟用Parquet存儲
        """
        self.symbols = symbols
        self.mt4_bridge = mt4_bridge or MT4Bridge()
        self.cache_size = cache_size
        self.storage_path = Path(storage_path)
        self.auto_save_interval = auto_save_interval
        self.enable_csv = enable_csv
        self.enable_parquet = enable_parquet

        # 創建存儲目錄
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "csv").mkdir(exist_ok=True)
        (self.storage_path / "parquet").mkdir(exist_ok=True)

        # 數據緩存 - 每個品種一個 deque
        self.tick_cache: Dict[str, deque] = {symbol: deque(maxlen=cache_size) for symbol in symbols}

        # 回調函數列表
        self.callbacks: List[Callable[[TickData], None]] = []

        # 控制標誌
        self._running = False
        self._collecting = False

        # 統計信息
        self.stats = {
            "total_ticks": 0,
            "ticks_per_symbol": defaultdict(int),
            "last_save_time": time.time(),
            "start_time": None,
        }

        # 線程池用於異步保存
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TickSaver")

        # 數據隊列用於異步處理
        self.save_queue = queue.Queue()
        self.save_thread = None

        logger.info(f"Tick收集器已初始化，監控品種: {symbols}")

    def add_callback(self, callback: Callable[[TickData], None]) -> None:
        """添加 Tick 數據回調函數"""
        self.callbacks.append(callback)
        logger.info(f"已添加回調函數: {callback.__name__}")

    def remove_callback(self, callback: Callable[[TickData], None]) -> None:
        """移除回調函數"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"已移除回調函數: {callback.__name__}")

    def _process_tick_data(self, raw_data: Dict[str, Any]) -> Optional[TickData]:
        """處理原始 Tick 數據"""
        try:
            if "symbol" not in raw_data or "bid" not in raw_data or "ask" not in raw_data:
                logger.warning(f"收到不完整的 Tick 數據: {raw_data}")
                return None

            tick = TickData(
                symbol=raw_data["symbol"],
                timestamp=datetime.now(timezone.utc),
                bid=float(raw_data["bid"]),
                ask=float(raw_data["ask"]),
                last=raw_data.get("last"),
                volume=int(raw_data.get("volume", 0)),
            )

            return tick

        except (ValueError, KeyError) as e:
            logger.error(f"處理 Tick 數據時出錯: {e}, 數據: {raw_data}")
            return None

    def _store_tick(self, tick: TickData) -> None:
        """存儲 Tick 數據到緩存"""
        symbol = tick.symbol
        if symbol in self.tick_cache:
            self.tick_cache[symbol].append(tick)

            # 更新統計
            self.stats["total_ticks"] += 1
            self.stats["ticks_per_symbol"][symbol] += 1

            # 觸發回調函數
            for callback in self.callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"回調函數 {callback.__name__} 執行失敗: {e}")

    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """獲取指定品種的最新 Tick"""
        if symbol in self.tick_cache and self.tick_cache[symbol]:
            return self.tick_cache[symbol][-1]
        return None

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[TickData]:
        """獲取指定品種的最近 Tick 數據"""
        if symbol not in self.tick_cache:
            return []

        cache = self.tick_cache[symbol]
        return list(cache)[-count:] if len(cache) >= count else list(cache)

    def get_tick_dataframe(self, symbol: str, count: int = None) -> pd.DataFrame:
        """獲取 Tick 數據的 DataFrame"""
        ticks = (
            self.get_recent_ticks(symbol, count) if count else list(self.tick_cache.get(symbol, []))
        )

        if not ticks:
            return pd.DataFrame(columns=["timestamp", "bid", "ask", "last", "volume"])

        data = [tick.to_dict() for tick in ticks]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    def _save_to_csv(self, symbol: str, ticks: List[TickData]) -> None:
        """保存數據到 CSV 文件"""
        if not self.enable_csv or not ticks:
            return

        try:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = self.storage_path / "csv" / f"{symbol}_ticks_{date_str}.csv"

            # 檢查文件是否存在，決定是否寫入標題
            file_exists = filename.exists()

            with open(filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["timestamp", "symbol", "bid", "ask", "last", "volume"])

                for tick in ticks:
                    writer.writerow(
                        [
                            tick.timestamp.isoformat(),
                            tick.symbol,
                            tick.bid,
                            tick.ask,
                            tick.last,
                            tick.volume,
                        ]
                    )

            logger.debug(f"已保存 {len(ticks)} 筆 {symbol} Tick 數據到 CSV")

        except Exception as e:
            logger.error(f"保存 CSV 文件時出錯: {e}")

    def _save_to_parquet(self, symbol: str, ticks: List[TickData]) -> None:
        """保存數據到 Parquet 文件"""
        if not self.enable_parquet or not ticks:
            return

        try:
            df = pd.DataFrame([tick.to_dict() for tick in ticks])
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            date_str = datetime.now().strftime("%Y%m%d")
            filename = self.storage_path / "parquet" / f"{symbol}_ticks_{date_str}.parquet"

            # 如果文件存在，追加數據
            if filename.exists():
                existing_df = pd.read_parquet(filename)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_parquet(filename, index=False)
            logger.debug(f"已保存 {len(ticks)} 筆 {symbol} Tick 數據到 Parquet")

        except Exception as e:
            logger.error(f"保存 Parquet 文件時出錯: {e}")

    def _save_worker(self) -> None:
        """異步保存工作線程"""
        while self._running or not self.save_queue.empty():
            try:
                # 從隊列獲取保存任務
                task = self.save_queue.get(timeout=1)
                symbol, ticks = task

                # 執行保存
                if self.enable_csv:
                    self._save_to_csv(symbol, ticks)
                if self.enable_parquet:
                    self._save_to_parquet(symbol, ticks)

                self.save_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"保存工作線程出錯: {e}")

    def save_cache_data(self, symbol: str = None) -> None:
        """保存緩存數據到磁盤"""
        symbols_to_save = [symbol] if symbol else self.symbols

        for sym in symbols_to_save:
            if sym in self.tick_cache:
                ticks = list(self.tick_cache[sym])
                if ticks:
                    # 將保存任務加入隊列
                    self.save_queue.put((sym, ticks))
                    logger.info(f"已將 {len(ticks)} 筆 {sym} 數據加入保存隊列")

        self.stats["last_save_time"] = time.time()

    async def start_collecting(self) -> None:
        """開始收集 Tick 數據"""
        if self._collecting:
            logger.warning("數據收集已在運行中")
            return

        self._running = True
        self._collecting = True
        self.stats["start_time"] = time.time()

        # 啟動保存工作線程
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        logger.info("開始收集 MT4 Tick 數據...")

        # 向 MT4 發送開始收集命令
        for symbol in self.symbols:
            self.mt4_bridge.send_command("START_TICK_STREAM", symbol=symbol)

        try:
            last_save = time.time()

            while self._collecting:
                try:
                    # 從 MT4 接收數據
                    raw_data = self.mt4_bridge.receive_data(timeout=1000)

                    if raw_data:
                        tick = self._process_tick_data(raw_data)
                        if tick and tick.symbol in self.symbols:
                            self._store_tick(tick)

                    # 定期保存數據
                    current_time = time.time()
                    if current_time - last_save >= self.auto_save_interval:
                        self.save_cache_data()
                        last_save = current_time

                    # 短暫休眠以避免過度消耗 CPU
                    await asyncio.sleep(0.001)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"收集 Tick 數據時出錯: {e}")
                    await asyncio.sleep(1)

        finally:
            # 停止收集時保存所有緩存數據
            logger.info("正在保存剩餘的 Tick 數據...")
            self.save_cache_data()

            # 向 MT4 發送停止命令
            for symbol in self.symbols:
                self.mt4_bridge.send_command("STOP_TICK_STREAM", symbol=symbol)

    def stop_collecting(self) -> None:
        """停止收集數據"""
        if not self._collecting:
            return

        logger.info("正在停止 Tick 數據收集...")
        self._collecting = False

        # 等待保存隊列完成
        if self.save_queue:
            self.save_queue.join()

        # 停止運行標誌
        self._running = False

        # 等待保存線程結束
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=5)

        logger.info("Tick 數據收集已停止")

    def get_statistics(self) -> Dict[str, Any]:
        """獲取收集統計信息"""
        current_time = time.time()
        runtime = current_time - self.stats["start_time"] if self.stats["start_time"] else 0

        return {
            "total_ticks": self.stats["total_ticks"],
            "ticks_per_symbol": dict(self.stats["ticks_per_symbol"]),
            "runtime_seconds": runtime,
            "average_ticks_per_second": self.stats["total_ticks"] / runtime if runtime > 0 else 0,
            "last_save_time": datetime.fromtimestamp(self.stats["last_save_time"]).isoformat(),
            "cache_sizes": {symbol: len(cache) for symbol, cache in self.tick_cache.items()},
            "is_collecting": self._collecting,
        }

    def clear_cache(self, symbol: str = None) -> None:
        """清空指定品種的緩存，如果未指定則清空所有"""
        if symbol:
            if symbol in self.tick_cache:
                self.tick_cache[symbol].clear()
                logger.info(f"已清空 {symbol} 的 Tick 緩存")
        else:
            for cache in self.tick_cache.values():
                cache.clear()
            self.stats["total_ticks"] = 0
            self.stats["ticks_per_symbol"].clear()
            logger.info("已清空所有 Tick 緩存")

    def __del__(self):
        """析構函數"""
        try:
            self.stop_collecting()
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Tick收集器析構時出錯: {e}")


# === 使用示例 ===


async def example_usage():
    """Tick 收集器使用示例"""

    # 設置日誌
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 定義要收集的品種
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]

    # 創建收集器
    collector = TickCollector(
        symbols=symbols,
        cache_size=5000,
        storage_path="./tick_data",
        auto_save_interval=60,  # 1分鐘自動保存
    )

    # 添加回調函數
    def on_tick_received(tick: TickData):
        print(f"收到 Tick: {tick.symbol} Bid:{tick.bid} Ask:{tick.ask} @{tick.timestamp}")

    collector.add_callback(on_tick_received)

    try:
        # 開始收集
        await collector.start_collecting()

    except KeyboardInterrupt:
        print("用戶中斷...")
    finally:
        # 停止收集
        collector.stop_collecting()

        # 打印統計信息
        stats = collector.get_statistics()
        print("\n=== 收集統計 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())
