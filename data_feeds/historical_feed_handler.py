import asyncio
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from core.event_types import MarketDataEvent

logger = logging.getLogger(__name__)

class HistoricalDataFeedHandler:
    """
    处理历史数据的处理器，从CSV文件读取数据并模拟实时数据流。
    """
    def __init__(self,
                 event_queue: asyncio.Queue,
                 csv_filepath: str,
                 symbol: str,
                 interval_seconds: float = 1.0):
        self.event_queue = event_queue
        self.csv_filepath = csv_filepath
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self._running = False
        self._data: Optional[pd.DataFrame] = None
        self._current_index = 0

    def _load_data(self) -> bool:
        """加载CSV数据文件"""
        try:
            self._data = pd.read_csv(self.csv_filepath)
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self._data.columns for col in required_columns):
                logger.error(f"CSV文件缺少必要的列: {required_columns}")
                return False
            return True
        except Exception as e:
            logger.error(f"加载CSV文件时出错: {e}")
            return False

    def start_feed(self):
        """同步方法，用于在线程中启动数据流"""
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start_feed())
        finally:
            loop.close()

    async def _async_start_feed(self):
        """异步方法，实际执行数据流模拟"""
        if not self._load_data():
            logger.error("无法加载数据，退出")
            return

        self._running = True
        logger.info(f"开始模拟历史数据流，共 {len(self._data)} 条记录")

        while self._running and self._current_index < len(self._data):
            try:
                row = self._data.iloc[self._current_index]
                
                # 创建市场数据事件
                market_event = MarketDataEvent(
                    symbol=self.symbol,
                    timestamp=datetime.now(timezone.utc),
                    event_type="AGG_MINUTE",
                    data={
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row['Volume'])
                    }
                )
                
                # 将事件放入队列（同步队列）
                self.event_queue.put(market_event)
                logger.debug(f"已发送历史数据事件: {market_event}")
                
                # 等待指定的时间间隔
                await asyncio.sleep(self.interval_seconds)
                self._current_index += 1
                
            except Exception as e:
                logger.error(f"处理历史数据时出错: {e}")
                break

        logger.info("历史数据流模拟结束")

    def stop(self):
        """停止数据流"""
        self._running = False
        logger.info("HistoricalDataFeedHandler 已停止") 