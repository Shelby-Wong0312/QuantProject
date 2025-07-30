# quant_project/main.py
# FINAL FIX - Corrected Strategy Name

import asyncio
import logging
import signal
import sys
from typing import List

import config
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("requests").setLevel(logging.WARNING)

from core.event_loop import EventLoop
from core.event import EventType, MarketEvent
from data_pipeline.live_feed import LiveDataFeed
from strategies.trading_strategies import ComprehensiveStrategy # <--- 修正名稱
from execution.portfolio import Portfolio
from execution.broker import Broker

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = asyncio.get_event_loop()
        self.event_queue = EventLoop()
        self.data_feed = None
        self.tasks: List[asyncio.Task] = []

    def run(self):
        logger.info("初始化 Capital.com 實時交易系統...")
        portfolio = Portfolio(self.event_queue)
        broker = Broker(self.event_queue)
        
        strategies = []
        for symbol in config.SYMBOLS_TO_TRADE:
            strategy_params = config.STRATEGY_PARAMS['Comprehensive_v1']
            strategy = ComprehensiveStrategy(symbol=symbol, params=strategy_params) # <--- 修正名稱
            strategies.append(strategy)
            logger.info(f"已為 {symbol} 創建策略實例。")

        self.event_queue.add_handler(EventType.FILL, portfolio.on_fill)
        self.event_queue.add_handler(EventType.ORDER, broker.on_order)
        self.event_queue.add_handler(EventType.SIGNAL, portfolio.on_signal)
        
        async def strategy_handler(event: MarketEvent):
            for s in strategies:
                if s.symbol == event.symbol:
                    signal = s.calculate_signals(event)
                    if signal:
                        await self.event_queue.put_event(signal)
                        break
        
        self.event_queue.add_handler(EventType.MARKET, strategy_handler)

        self.data_feed = LiveDataFeed(symbols=config.SYMBOLS_TO_TRADE, event_queue=self.event_queue)
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except (NotImplementedError, RuntimeError):
                pass

        try:
            logger.info("交易系統啟動完成。 按下 Ctrl+C 來關閉。")
            self.tasks.append(self.loop.create_task(self.event_queue.run()))
            self.tasks.append(self.loop.create_task(self.data_feed.run()))
            self.loop.run_forever()
        finally:
            self.loop.close()

    async def shutdown(self):
        logger.info("收到關閉訊號，開始關閉系統...")
        if self.data_feed: self.data_feed.stop()
        for task in self.tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.loop.is_running(): self.loop.stop()
        logger.info("系統已成功關閉。")

if __name__ == "__main__":
    system = TradingSystem()
    system.run()