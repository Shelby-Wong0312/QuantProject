# strategy/base.py
import abc
from typing import List
from core.event import MarketDataEvent, SignalEvent

class AbstractStrategy(abc.ABC):
    """
    所有策略類別的抽象基類。
    """
    @abc.abstractmethod
    async def on_data(self, event: MarketDataEvent) -> List[SignalEvent]:
        """
        處理傳入的市場數據事件並返回一個信號事件列表。
        即使不產生信號，也應返回一個空列表。
        """
        raise NotImplementedError("Should implement on_data()")