# strategy/stateful_strategy.py
import logging
from typing import List, Dict, Any
from core.event import MarketDataEvent, SignalEvent, SignalAction
from strategy.base import AbstractStrategy

logger = logging.getLogger(__name__)

class StatefulStrategy(AbstractStrategy):
    """
    一個能為多個股票維護獨立狀態，並產生交易信號的策略。
    """
    def __init__(self, symbols_to_manage: List[str]):
        self.symbols_to_manage = symbols_to_manage
        self.symbol_states: Dict[str, Dict[str, Any]] = {}
        
        # 初始化每個股票的狀態
        for symbol in self.symbols_to_manage:
            self.symbol_states[symbol] = {
                'data_count': 0,
                'signal_sent': False # 新增一個旗標，避免重複發送信號
            }
        logger.info(f"StatefulStrategy initialized for symbols: {self.symbols_to_manage}")

    async def on_data(self, event: MarketDataEvent) -> List[SignalEvent]:
        """
        當收到市場數據時，更新狀態，並在滿足條件時產生信號。
        """
        symbol = event.symbol
        if symbol not in self.symbol_states:
            return []

        state = self.symbol_states[symbol]
        
        # 更新狀態
        state['data_count'] += 1
        count = state['data_count']
        logger.info(f"StatefulStrategy state for {symbol}: data_count is now {count}")
        
        # --- 新增：決策與信號產生邏輯 ---
        # 如果數據計數達到 5，且尚未發送過信號
        if count == 5 and not state['signal_sent']:
            logger.info(f"CONDITION MET! Generating BUY signal for {symbol}")
            
            # 建立一個 BUY 信號事件
            signal = SignalEvent(
                symbol=symbol,
                action=SignalAction.BUY,
                quantity=100 # 假設數量
            )
            
            # 更新旗標，防止再次發送
            state['signal_sent'] = True
            
            # 返回包含信號的列表
            return [signal]
        
        # 如果不滿足條件，返回空列表
        return []