# risk_management/risk_manager.py

import asyncio
import logging
from typing import Optional, Dict

from core.event_types import SignalEvent, OrderEvent, FillEvent
from core import config
from core import utils # <--- 修正引用方式

logger = logging.getLogger(__name__)

class RiskManager:
    # ... (類別內部程式碼不變，僅修正引用後的函式調用) ...
    def __init__(self,
                 signal_queue: asyncio.Queue,
                 order_queue: asyncio.Queue,
                 fill_queue: Optional[asyncio.Queue] = None):
        self.signal_queue = signal_queue
        self.order_queue = order_queue
        self.fill_queue = fill_queue
        self.max_order_qty = config.MAX_ORDER_QUANTITY_PER_TRADE
        self.max_pos_value = config.MAX_POSITION_VALUE_PER_SYMBOL
        self.current_positions: Dict[str, Dict] = {}
        self._running = False

    async def _check_risk(self, signal: SignalEvent) -> Optional[OrderEvent]:
        order_quantity = signal.target_quantity if signal.target_quantity is not None else self.max_order_qty
        if order_quantity > self.max_order_qty:
            logger.warning(f"Risk Check FAILED for signal {signal.strategy_id} on {signal.symbol}: "
                           f"Proposed quantity {order_quantity} > max_order_qty {self.max_order_qty}. Reducing.")
            order_quantity = self.max_order_qty
        if order_quantity <= 0:
            logger.warning(f"Risk Check FAILED for signal {signal.strategy_id} on {signal.symbol}: "
                           f"Proposed quantity {order_quantity} is not positive. Rejecting.")
            return None
        estimated_price = signal.limit_price
        if signal.order_type == "MARKET" and estimated_price is None:
            pass
        elif estimated_price:
            order_value = order_quantity * estimated_price
            if order_value > self.max_pos_value:
                logger.warning(f"Risk Check FAILED for signal {signal.strategy_id} on {signal.symbol}: "
                               f"Order value {order_value:.2f} > max_pos_value_per_order {self.max_pos_value}. Rejecting.")
                return None
        # 使用 utils.generate_client_order_id() 和 utils.get_current_timestamp()
        client_order_id = utils.generate_client_order_id()
        order_event = OrderEvent(
            timestamp=utils.get_current_timestamp(),
            symbol=signal.symbol,
            order_type=signal.order_type,
            direction=signal.direction,
            quantity=order_quantity,
            limit_price=signal.limit_price,
            client_order_id=client_order_id,
            strategy_id=signal.strategy_id,
            notes="Risk Approved"
        )
        logger.info(f"Risk Check PASSED for signal {signal.strategy_id} on {signal.symbol}. Order: {order_event}")
        return order_event

    async def _handle_signal_queue(self, shutdown_event: asyncio.Event):
        while self._running and not shutdown_event.is_set():
            try:
                signal_event: SignalEvent = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                order_event = await self._check_risk(signal_event)
                if order_event:
                    await self.order_queue.put(order_event)
                self.signal_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"RiskManager: Error processing signal: {e}", exc_info=True)

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("RiskManager starting...")
        tasks = [self._handle_signal_queue(shutdown_event)]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if not shutdown_event.is_set():
            logger.warning("RiskManager: A sub-task finished unexpectedly. Initiating stop.")
        await self.stop()
        logger.info("RiskManager stopped.")
        
    async def stop(self):
        self._running = False
        logger.info("RiskManager stopping...")