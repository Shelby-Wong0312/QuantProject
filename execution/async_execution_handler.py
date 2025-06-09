# execution/async_execution_handler.py
import asyncio
import logging
from datetime import datetime, timezone
from core.event import OrderEvent, FillEvent
# from execution.capital_client import AsyncCapitalComClient # 假設的券商客戶端

logger = logging.getLogger(__name__)

# --- 暫時代替的模擬券商客戶端 ---
class MockBrokerClient:
    async def place_market_order(self, symbol, action, quantity):
        logger.info(f"MOCK BROKER: Placing market order for {quantity} {symbol} ({action.value})")
        # 模擬API調用延遲
        await asyncio.sleep(0.1)
        # 模擬成交回報
        mock_fill_price = 150.0 # 假設的成交價
        logger.info(f"MOCK BROKER: Order for {symbol} filled at ${mock_fill_price}")
        return True, mock_fill_price
# --- 結束模擬 ---

class AsyncExecutionHandler:
    def __init__(self, event_queue_in: asyncio.Queue, event_queue_out: asyncio.Queue):
        self.event_queue_in = event_queue_in   # For OrderEvent
        self.event_queue_out = event_queue_out # For FillEvent
        # self.broker_client = broker_client # 接收一個券商客戶端實例
        self.broker_client = MockBrokerClient() # <-- 使用模擬客戶端
        self.running = False
        self._processing_task = None

    async def _process_events(self):
        logger.info("Execution Handler event processing started.")
        self.running = True
        while self.running:
            try:
                order: OrderEvent = await asyncio.wait_for(self.event_queue_in.get(), timeout=1.0)
                
                # 調用券商客戶端下單
                success, fill_price = await self.broker_client.place_market_order(
                    symbol=order.symbol,
                    action=order.action,
                    quantity=order.quantity
                )

                if success:
                    # 產生一個成交事件
                    fill_event = FillEvent(
                        symbol=order.symbol,
                        timestamp=datetime.now(timezone.utc),
                        action=order.action,
                        quantity=order.quantity,
                        fill_price=fill_price
                    )
                    await self.event_queue_out.put(fill_event)

                self.event_queue_in.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in Execution Handler event processing: {e}", exc_info=True)

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
            logger.info("Execution Handler stopped.")