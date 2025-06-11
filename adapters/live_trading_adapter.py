# 檔案位置: adapters/live_trading_adapter.py

import logging
import pandas as pd

from strategy.abstract_strategy import AbstractStrategyBase
# 導入我們定義的事件
from core.event import MarketDataEvent, SignalEvent

logger = logging.getLogger(__name__)

class LiveTradingAdapter:
    """
    一個事件驅動的適配器。
    它「消費」市場數據事件，並「產生」交易訊號事件。
    """
    def __init__(self, event_queue, abstract_strategy: AbstractStrategyBase):
        """
        現在它需要 event_queue 來發布新的事件。
        注意：它不再需要 capital_client，因為它不直接負責交易。
        """
        self.event_queue = event_queue
        self.abstract_strategy = abstract_strategy

    async def handle_market_data_event(self, event: MarketDataEvent):
        """
        這是一個事件處理器，將在 EventLoop 中註冊。
        它處理市場數據並產生交易訊號。
        """
        # 檢查數據標的
        if self.abstract_strategy.symbol != event.symbol:
            return

        # 1. 呼叫策略，獲取交易意圖 (返回的是 Signal 物件)
        signals_to_execute = self.abstract_strategy.on_data(event.ohlcv_data)
        
        if not signals_to_execute:
            return

        # 2. 將策略的「意圖」轉換為標準化的「訊號事件」
        for signal in signals_to_execute:
            logger.info(f"適配器從策略獲取訊號: {signal.action} for {signal.symbol}")
            
            # 建立一個 SignalEvent
            signal_event = SignalEvent(
                strategy_id=type(self.abstract_strategy).__name__,
                symbol=signal.symbol,
                action=signal.action,
                quantity=self.abstract_strategy.parameters.get('live_trade_quantity', 0.01),
                price=signal.price,
                sl_price=signal.sl,
                tp_price=signal.tp,
                comment=signal.comment,
                correlation_id=event.event_id # 追蹤事件來源
            )
            
            # 3. 將訊號事件發布回事件隊列，交由其他模組處理
            self.event_queue.put(signal_event)
            