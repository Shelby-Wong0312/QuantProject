# strategy/stateful_strategy.py
import logging
from typing import List, Dict, Any
from core.event import MarketDataEvent, SignalEvent
from strategy.base import AbstractStrategy

logger = logging.getLogger(__name__)

class StatefulStrategy(AbstractStrategy):
    """
    一個能為多個股票維護獨立狀態的策略。
    """
    def __init__(self, symbols_to_manage: List[str]):
        self.symbols_to_manage = symbols_to_manage
        self.symbol_states: Dict[str, Dict[str, Any]] = {}
        
        # 初始化每個股票的狀態
        for symbol in self.symbols_to_manage:
            self.symbol_states[symbol] = {
                'data_count': 0
            }
        logger.info(f"StatefulStrategy initialized for symbols: {self.symbols_to_manage}")

    async def on_data(self, event: MarketDataEvent) -> List[SignalEvent]:
        """
        當收到市場數據時，更新對應股票的狀態並印出日誌。
        """
        symbol = event.symbol
        if symbol not in self.symbol_states:
            return []

        # 更新狀態
        self.symbol_states[symbol]['data_count'] += 1
        
        # 打印日誌以驗證狀態管理
        count = self.symbol_states[symbol]['data_count']
        logger.info(f"StatefulStrategy state for {symbol}: data_count is now {count}")
        
        return []