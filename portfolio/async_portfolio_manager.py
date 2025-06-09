# portfolio/async_portfolio_manager.py
import asyncio
import logging
from collections import defaultdict
from core.event import FillEvent, SignalAction

logger = logging.getLogger(__name__)

class AsyncPortfolioManager:
    def __init__(self, event_queue_in: asyncio.Queue, initial_cash: float = 100000.0):
        self.event_queue_in = event_queue_in # For FillEvent
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        
        # 用來儲存每個 symbol 的持倉數量
        self.positions = defaultdict(float)
        
        self.running = False
        self._processing_task = None

    def _update_position_from_fill(self, fill: FillEvent):
        """根據成交事件來更新持倉和現金。"""
        symbol = fill.symbol
        quantity = fill.quantity
        fill_price = fill.fill_price
        
        # 計算持倉變化
        if fill.action == SignalAction.BUY:
            self.positions[symbol] += quantity
            self.current_cash -= quantity * fill_price
        elif fill.action == SignalAction.SELL:
            self.positions[symbol] -= quantity
            self.current_cash += quantity * fill_price
        
        logger.info(f"PORTFOLIO UPDATE: Position for {symbol} is now {self.positions[symbol]}. Current cash: ${self.current_cash:,.2f}")

    async def _process_events(self):
        logger.info("Portfolio Manager event processing started.")
        self.running = True
        while self.running:
            try:
                fill: FillEvent = await asyncio.wait_for(self.event_queue_in.get(), timeout=1.0)
                self._update_position_from_fill(fill)
                self.event_queue_in.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in Portfolio Manager event processing: {e}", exc_info=True)

    def start(self):
        if not self.running:
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self):
        if self.running:
            self.running = False
            if self._processing_task:
                self._processing_task.cancel()
                try: await self._processing_task
                except asyncio.CancelledError: pass
            logger.info("Portfolio Manager stopped.")