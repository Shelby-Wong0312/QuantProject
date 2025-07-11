# quant_project/execution/portfolio.py
# FINAL FIX - Missing Import

import asyncio
import logging
from collections import defaultdict
from datetime import datetime

from core.event import SignalEvent, OrderEvent, FillEvent
from core.event_loop import EventLoop
import config # <--- 新增這一行，解決 NameError

logger = logging.getLogger(__name__)

class Portfolio:
    """
    投資組合與風險管理模組。
    接收交易信號，執行風險檢查，管理倉位和資金。
    """
    def __init__(self, event_queue: EventLoop):
        self.event_queue = event_queue
        self.initial_capital = config.INITIAL_CAPITAL
        self.current_cash = config.INITIAL_CAPITAL
        self.positions = defaultdict(float) # {'AAPL.US': 10}
        self.trades = []

    async def on_signal(self, signal: SignalEvent):
        """
        接收信號並執行風險檢查，通過後生成訂單事件。
        """
        logger.info(f"📈 [Portfolio] 收到信號: {signal.direction} {signal.quantity} {signal.symbol}")

        # 基本風險檢查
        if signal.quantity <= 0:
            logger.warning(f"風險檢查失敗: {signal.symbol} 交易數量 ({signal.quantity}) 無效。")
            return
        
        logger.info(f"✅ [Portfolio] 風險檢查通過，為 {signal.symbol} 生成訂單。")
        order = OrderEvent(
            symbol=signal.symbol,
            timestamp=datetime.now(),
            direction=signal.direction,
            quantity=signal.quantity
        )
        await self.event_queue.put_event(order)

    async def on_fill(self, fill: FillEvent):
        """
        根據成交事件更新持倉和現金。
        """
        logger.info(f"🧾 [Portfolio] 更新資產: {fill.direction} {fill.quantity} {fill.symbol} @ {fill.fill_price:.2f}")
        trade_value = fill.quantity * fill.fill_price

        if fill.direction.upper() == 'BUY':
            self.current_cash -= trade_value
            self.positions[fill.symbol] += fill.quantity
        elif fill.direction.upper() == 'SELL':
            self.current_cash += trade_value
            self.positions[fill.symbol] -= fill.quantity

        if self.positions[fill.symbol] == 0:
            del self.positions[fill.symbol]
            
        self.trades.append(fill)
        self.log_portfolio_status()

    def log_portfolio_status(self):
        """打印當前的投資組合狀態。"""
        print("-" * 50)
        logger.info("====== 投資組合狀態 ======")
        logger.info(f"💰 當前現金: ${self.current_cash:,.2f}")
        if self.positions:
            logger.info("📊 當前持倉:")
            for symbol, quantity in self.positions.items():
                logger.info(f"   - {symbol}: {quantity} 股")
        else:
            logger.info("📊 當前無持倉。")
        logger.info("==========================")
        print("-" * 50)