import asyncio
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional

from core.event_types import FillEvent, MarketDataEvent
from core import config

logger = logging.getLogger(__name__)

class Position:
    """
    代表一個特定資產的持倉狀態。
    使用 Decimal 來進行精確的財務計算。
    """
    def __init__(self, symbol: str, quantity: Decimal = Decimal("0"),
                 average_entry_price: Decimal = Decimal("0")):
        self.symbol = symbol
        self.quantity = quantity
        self.average_entry_price = average_entry_price
        self.realized_pnl = Decimal("0")
        self.last_market_price = Decimal("0")

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.quantity == 0:
            return Decimal("0")
        return (self.last_market_price - self.average_entry_price) * self.quantity

    def update_on_fill(self, fill: FillEvent):
        if fill.fill_quantity is None or fill.fill_price is None:
            return

        fill_qty = Decimal(str(fill.fill_quantity))
        fill_price = Decimal(str(fill.fill_price))
        
        if fill.direction == "BUY":
            # 計算已實現盈虧 (如果是在回補空頭倉位)
            if self.quantity < 0:
                pnl_from_cover = (self.average_entry_price - fill_price) * min(abs(self.quantity), fill_qty)
                self.realized_pnl += pnl_from_cover

            # 更新平均成本
            new_total_value = (self.quantity * self.average_entry_price) + (fill_qty * fill_price)
            self.quantity += fill_qty
            if self.quantity != 0:
                self.average_entry_price = new_total_value / self.quantity
            else:
                self.average_entry_price = Decimal("0")

        elif fill.direction == "SELL":
            # 計算已實現盈虧 (如果是在賣出多頭倉位)
            if self.quantity > 0:
                pnl_from_sale = (fill_price - self.average_entry_price) * min(self.quantity, fill_qty)
                self.realized_pnl += pnl_from_sale

            # 更新平均成本 (對於做空，成本計算可能更複雜，此處為簡化模型)
            self.quantity -= fill_qty
            if self.quantity == 0:
                self.average_entry_price = Decimal("0")

        logger.info(f"Position updated for {self.symbol}: Qty={self.quantity}, AvgPx={self.average_entry_price:.4f}, RealizedPNL={self.realized_pnl:.2f}")

    def update_market_price(self, price: float):
        self.last_market_price = Decimal(str(price))


class PortfolioManager:
    """
    管理整個投資組合的狀態，包括現金、持倉和盈虧。
    """
    def __init__(self,
                 fill_queue: asyncio.Queue,
                 market_data_queue: Optional[asyncio.Queue] = None):
        self.fill_queue = fill_queue
        self.market_data_queue = market_data_queue
        
        initial_capital = config.INITIAL_CAPITAL_EXAMPLE
        self.cash: Decimal = Decimal(str(initial_capital))
        self.positions: Dict[str, Position] = {}
        self._running = False
    
    def _get_or_create_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    async def _handle_fill_event(self, fill_event: FillEvent):
        if fill_event.status not in ["FILLED", "PARTIALLY_FILLED"] or \
           fill_event.fill_quantity is None or fill_event.fill_price is None:
            logger.debug(f"PortfolioManager ignoring non-fill or incomplete FillEvent: {fill_event.client_order_id}")
            return

        logger.debug(f"PortfolioManager processing fill: {fill_event}")
        position = self._get_or_create_position(fill_event.symbol)
        
        fill_value = Decimal(str(fill_event.fill_quantity)) * Decimal(str(fill_event.fill_price))
        commission = Decimal(str(fill_event.commission))
        
        if fill_event.direction == "BUY":
            self.cash -= (fill_value + commission)
        elif fill_event.direction == "SELL":
            self.cash += (fill_value - commission)
            
        position.update_on_fill(fill_event)
        
        logger.info(f"Portfolio state: Cash={self.cash:.2f}, Positions Count={len(self.positions)}")

    async def _handle_market_data_event(self, market_event: MarketDataEvent):
        """處理市場數據以更新未實現盈虧。"""
        if market_event.symbol in self.positions and market_event.price is not None:
            position = self.positions[market_event.symbol]
            position.update_market_price(market_event.price)
            logger.debug(f"Updated UPL for {market_event.symbol}: {position.unrealized_pnl:.2f}")

    async def _fill_queue_loop(self, shutdown_event: asyncio.Event):
        while self._running and not shutdown_event.is_set():
            try:
                fill_event: FillEvent = await asyncio.wait_for(self.fill_queue.get(), timeout=1.0)
                await self._handle_fill_event(fill_event)
                self.fill_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"PortfolioManager: Error processing fill event: {e}", exc_info=True)

    async def _market_data_queue_loop(self, shutdown_event: asyncio.Event):
        if not self.market_data_queue:
            return
        while self._running and not shutdown_event.is_set():
            try:
                market_event: MarketDataEvent = await asyncio.wait_for(self.market_data_queue.get(), timeout=1.0)
                await self._handle_market_data_event(market_event)
                self.market_data_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"PortfolioManager: Error processing market data event: {e}", exc_info=True)

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("PortfolioManager starting...")
        
        tasks = [self._fill_queue_loop(shutdown_event)]
        if self.market_data_queue:
            tasks.append(self._market_data_queue_loop(shutdown_event))
            
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for task in pending:
            task.cancel()
            
        if not shutdown_event.is_set():
            logger.warning("PortfolioManager: A sub-task finished unexpectedly. Initiating stop.")

        await self.stop()
        logger.info("PortfolioManager stopped.")

    async def stop(self):
        self._running = False
        logger.info("PortfolioManager stopping...")
        logger.info(f"Final portfolio: Cash={self.cash:.2f}, Positions={self.positions}")