# execution/execution_handler.py

import asyncio
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OrderSide, TimeInForce
from alpaca.trading.enums import OrderType
from alpaca.trading.models import TradeUpdate, Order as AlpacaOrder

from core.event_types import OrderEvent, FillEvent
from core import config
from core.utils import get_current_timestamp

logger = logging.getLogger(__name__)

class ExecutionHandler:
    """
    處理訂單執行，與 Alpaca API 交互，並產生 FillEvent。
    """
    def __init__(self,
                 order_queue: asyncio.Queue,
                 fill_queue: asyncio.Queue,
                 api_key: str = config.ALPACA_API_KEY_ID,
                 secret_key: str = config.ALPACA_SECRET_KEY,
                 paper: bool = config.ALPACA_PAPER_TRADING):
        self.order_queue = order_queue
        self.fill_queue = fill_queue
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper

        self.trading_client = TradingClient(self._api_key, self._secret_key, paper=self._paper)
        self.trading_stream = TradingStream(self._api_key, self._secret_key, paper=self._paper)

        self._running = False
        self._active_orders = {}  # client_order_id -> OrderEvent

    async def _on_trade_update(self, trade_update_data: TradeUpdate):
        """處理來自 Alpaca WebSocket 的交易更新。"""
        try:
            logger.info(f"ExecutionHandler received trade update: {trade_update_data}")
            
            # 將 Alpaca 的事件類型映射到我們的 FillEvent 狀態
            status_mapping = {
                "new": "NEW", "fill": "FILLED", "partial_fill": "PARTIALLY_FILLED",
                "canceled": "CANCELED", "rejected": "REJECTED", "expired": "EXPIRED",
                "done_for_day": "FILLED", "pending_new": "NEW"
            }
            order_data = trade_update_data.order
            fill_event_status = status_mapping.get(trade_update_data.event, "UNKNOWN_STATUS")

            client_order_id = order_data.client_order_id
            if not client_order_id:
                logger.error(f"TradeUpdate missing client_order_id: {trade_update_data}")
                return

            fill_event = FillEvent(
                timestamp=order_data.updated_at or get_current_timestamp(),
                symbol=order_data.symbol,
                client_order_id=client_order_id,
                broker_order_id=str(order_data.id),
                fill_id=str(trade_update_data.id) if trade_update_data.event in ["fill", "partial_fill"] else str(order_data.id),
                status=fill_event_status,
                direction=str(order_data.side).upper(),
                fill_price=float(trade_update_data.price) if trade_update_data.price else None,
                fill_quantity=int(trade_update_data.qty) if trade_update_data.qty else None,
                cumulative_filled_quantity=int(order_data.filled_qty),
                average_fill_price=float(order_data.filled_avg_price) if order_data.filled_avg_price else None,
                commission=float(trade_update_data.order.get('commission', 0.0)),
                remaining_quantity=int(order_data.qty) - int(order_data.filled_qty)
            )

            await self.fill_queue.put(fill_event)

            # 如果訂單已終結，從活動訂單中移除
            if fill_event_status in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                if client_order_id in self._active_orders:
                    del self._active_orders[client_order_id]
                    logger.info(f"Order {client_order_id} removed from active orders due to status: {fill_event_status}")

        except Exception as e:
            logger.error(f"Error processing trade update: {trade_update_data}. Error: {e}", exc_info=True)

    async def _submit_order(self, order_event: OrderEvent):
        """向 Alpaca API 提交訂單。"""
        try:
            order_data = None
            if order_event.order_type.upper() == "MARKET":
                order_data = MarketOrderRequest(
                    symbol=order_event.symbol,
                    qty=order_event.quantity,
                    side=OrderSide(order_event.direction.lower()),
                    time_in_force=TimeInForce.GTC,
                    client_order_id=order_event.client_order_id
                )
            elif order_event.order_type.upper() == "LIMIT":
                if order_event.limit_price is None:
                    raise ValueError("Limit price must be set for a LIMIT order.")
                order_data = LimitOrderRequest(
                    symbol=order_event.symbol,
                    qty=order_event.quantity,
                    side=OrderSide(order_event.direction.lower()),
                    time_in_force=TimeInForce.GTC,
                    limit_price=order_event.limit_price,
                    client_order_id=order_event.client_order_id
                )

            if order_data:
                submitted_alpaca_order = self.trading_client.submit_order(order_data=order_data)
                self._active_orders[submitted_alpaca_order.client_order_id] = order_event
                logger.info(f"Order submitted to Alpaca: {submitted_alpaca_order}. Client Order ID: {order_event.client_order_id}")
            else:
                raise ValueError(f"Unsupported order type: {order_event.order_type}")

        except Exception as e:
            logger.error(f"Error submitting order {order_event.client_order_id} for {order_event.symbol}: {e}", exc_info=True)
            # 創建一個 REJECTED FillEvent
            rejected_fill = FillEvent(
                timestamp=get_current_timestamp(),
                symbol=order_event.symbol,
                client_order_id=order_event.client_order_id,
                broker_order_id="N/A",
                fill_id=order_event.client_order_id,
                status="REJECTED",
                direction=order_event.direction,
                message=f"Submission failed: {str(e)}"
            )
            await self.fill_queue.put(rejected_fill)

    async def _order_submission_loop(self, shutdown_event: asyncio.Event):
        """處理內部訂單佇列。"""
        while self._running and not shutdown_event.is_set():
            try:
                order_event: OrderEvent = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                await self._submit_order(order_event)
                self.order_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"ExecutionHandler: Error in order submission loop: {e}", exc_info=True)

    async def _trade_update_loop(self, shutdown_event: asyncio.Event):
        """保持交易更新流的訂閱和活動狀態。"""
        self.trading_stream.subscribe_trade_updates(self._on_trade_update)
        logger.info("ExecutionHandler: Subscribed to Alpaca trade updates.")
        while self._running and not shutdown_event.is_set():
            await asyncio.sleep(1) # 讓出控制權，實際處理在 _on_trade_update 中
        logger.info("ExecutionHandler: Trade update loop is shutting down.")

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("ExecutionHandler starting...")
        
        order_submit_task = asyncio.create_task(self._order_submission_loop(shutdown_event))
        trade_update_task = asyncio.create_task(self.trading_stream.run()) # TradingStream.run() 處理連接
        
        tasks = [order_submit_task, trade_update_task]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for task in pending:
            task.cancel()
            
        if not shutdown_event.is_set():
            logger.warning("ExecutionHandler: A sub-task finished unexpectedly. Initiating stop.")

        await self.stop()
        logger.info("ExecutionHandler stopped.")

    async def stop(self):
        self._running = False
        logger.info("ExecutionHandler stopping...")
        if self._active_orders:
            logger.warning(f"ExecutionHandler stopping with {len(self._active_orders)} active orders: {list(self._active_orders.keys())}")
        
        try:
            if self.trading_stream:
                await self.trading_stream.close()
                logger.info("ExecutionHandler: Alpaca trading stream closed.")
        except Exception as e:
            logger.error(f"ExecutionHandler: Error during trading stream stop: {e}", exc_info=True)