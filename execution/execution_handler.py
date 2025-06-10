# execution/execution_handler.py
import asyncio
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OrderSide, TimeInForce
from alpaca.trading.models import TradeUpdate
from core.event_types import OrderEvent, FillEvent
from core import config
from core import utils

logger = logging.getLogger(__name__)

class ExecutionHandler:
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
        self._active_orders = {}

    async def _on_trade_update(self, trade_update_data: TradeUpdate):
        try:
            logger.info(f"ExecutionHandler received trade update: {trade_update_data}")
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
                symbol=order_data.symbol,
                timestamp=order_data.updated_at or utils.get_current_timestamp(),
                exchange="ALPACA", # Assuming Alpaca as the exchange
                quantity=float(trade_update_data.qty) if trade_update_data.qty else 0.0,
                fill_price=float(trade_update_data.price) if trade_update_data.price else 0.0,
                commission=float(trade_update_data.order.get('commission', 0.0)),
                order_id=str(order_data.id),
                direction=str(order_data.side).upper(),
                client_order_id=client_order_id,
                strategy_id=None # This would need to be retrieved from active_orders if needed
            )
            await self.fill_queue.put(fill_event)
            if fill_event_status in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                if client_order_id in self._active_orders:
                    del self._active_orders[client_order_id]
                    logger.info(f"Order {client_order_id} removed from active orders due to status: {fill_event_status}")
        except Exception as e:
            logger.error(f"Error processing trade update: {trade_update_data}. Error: {e}", exc_info=True)

    async def _submit_order(self, order_event: OrderEvent):
        try:
            order_data = None
            common_params = {
                "symbol": order_event.symbol,
                "qty": order_event.quantity,
                "side": OrderSide(order_event.side.lower()),
                "time_in_force": TimeInForce(order_event.time_in_force.lower())
            }
            if order_event.order_type.upper() == "MARKET":
                order_data = MarketOrderRequest(**common_params)
            elif order_event.order_type.upper() == "LIMIT":
                if order_event.limit_price is None:
                    raise ValueError("Limit price must be set for a LIMIT order.")
                order_data = LimitOrderRequest(**common_params, limit_price=order_event.limit_price)
            
            if order_data:
                # Add client_order_id if it exists
                if order_event.client_order_id:
                    order_data.client_order_id = order_event.client_order_id
                
                submitted_alpaca_order = self.trading_client.submit_order(order_data=order_data)
                self._active_orders[submitted_alpaca_order.client_order_id] = order_event
                logger.info(f"Order submitted to Alpaca: {submitted_alpaca_order}")
            else:
                raise ValueError(f"Unsupported order type: {order_event.order_type}")

        except Exception as e:
            logger.error(f"Error submitting order for {order_event.symbol}: {e}", exc_info=True)
            # Create a REJECTED FillEvent - implementation can be added here
            
    async def _order_submission_loop(self, shutdown_event: asyncio.Event):
        while self._running and not shutdown_event.is_set():
            try:
                order_event: OrderEvent = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                await self._submit_order(order_event)
                self.order_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"ExecutionHandler: Error in order submission loop: {e}", exc_info=True)

    # vvvvvv 修正此處的 run 方法 vvvvvv
    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("ExecutionHandler starting...")
        
        self.trading_stream.subscribe_trade_updates(self._on_trade_update)
        
        order_submit_task = asyncio.create_task(self._order_submission_loop(shutdown_event))
        
        loop = asyncio.get_running_loop()
        # 直接將 run_in_executor 回傳的 Future 物件用於 await，而不是再包一層 create_task
        trade_update_future = loop.run_in_executor(None, self.trading_stream.run)
        
        # asyncio.wait 可以同時處理 Task 和 Future 物件
        tasks_to_wait = [order_submit_task, trade_update_future]
        done, pending = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)
        
        for task in pending:
            task.cancel()
            
        if not shutdown_event.is_set():
            logger.warning("ExecutionHandler: A sub-task finished unexpectedly. Initiating stop.")

        await self.stop()
        logger.info("ExecutionHandler stopped.")
    # ^^^^^^ 修正此處的 run 方法 ^^^^^^

    async def stop(self):
        self._running = False
        logger.info("ExecutionHandler stopping...")
        if self._active_orders:
            logger.warning(f"ExecutionHandler stopping with {len(self._active_orders)} active orders.")
        
        try:
            if self.trading_stream:
                self.trading_stream.close()
                logger.info("ExecutionHandler: Alpaca trading stream closed.")
        except Exception as e:
            logger.error(f"ExecutionHandler: Error during trading stream stop: {e}", exc_info=True)