# strategy/logging_strategy.py
import logging
from typing import List
from core.event import MarketDataEvent, SignalEvent
from strategy.base import AbstractStrategy

logger = logging.getLogger(__name__)

class LoggingStrategy(AbstractStrategy):
    """
    一個簡單的策略，僅用於記錄接收到的市場數據事件。
    """
    async def on_data(self, event: MarketDataEvent) -> List[SignalEvent]:
        """
        當收到市場數據時，印出一條日誌，不產生任何交易信號。
        """
        logger.info(f"LoggingStrategy received data: {event.symbol} at {event.timestamp}")
        # 不產生任何信號，返回空列表
        return []