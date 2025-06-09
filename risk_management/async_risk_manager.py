# risk_management/async_risk_manager.py
import asyncio
import logging
from core.event import SignalEvent, OrderEvent, OrderType

logger = logging.getLogger(__name__)

class AsyncRiskManager:
    def __init__(self, event_queue_in: asyncio.Queue, event_queue_out: asyncio.Queue):
        self.event_queue_in = event_queue_in   # For SignalEvent
        self.event_queue_out = event_queue_out # For OrderEvent
        self.running = False
        self._processing_task = None
        
        # 簡單的風險規則：黑名單
        self.blacklist = ["T.GOOG"]

    async def _check_risk(self, signal: SignalEvent) -> bool:
        """
        檢查交易信號是否符合風險規則。
        返回 True 代表通過，False 代表否決。
        """
        if signal.symbol in self.blacklist:
            logger.warning(f"RISK VETOED: Signal {signal} is on the blacklist.")
            return False
        
        # 之後可以加入更多檢查，例如倉位大小、下單頻率等
        
        logger.info(f"RISK PASSED: Signal {signal} approved.")
        return True

    async def _process_events(self):
        logger.info("Risk Manager event processing started.")
        self.running = True
        while self.running:
            try:
                signal: SignalEvent = await asyncio.wait_for(self.event_queue_in.get(), timeout=1.0)
                
                is_approved = await self._check_risk(signal)
                
                if is_approved:
                    # 風險通過，將信號轉換為訂單事件
                    order = OrderEvent(
                        symbol=signal.symbol,
                        order_type=OrderType.MARKET, # 假設為市價單
                        action=signal.action,
                        quantity=signal.quantity
                    )
                    await self.event_queue_out.put(order)
                
                self.event_queue_in.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in Risk Manager event processing: {e}", exc_info=True)

    def start(self):
        if not self.running:
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self):
        if self.running:
            self.running = False
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    logger.info("Risk Manager processing task cancelled.")
            logger.info("Risk Manager stopped.")