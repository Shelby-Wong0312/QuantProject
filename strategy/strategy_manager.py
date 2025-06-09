# strategy/strategy_manager.py
import asyncio
import logging
from core.event import MarketDataEvent, SignalEvent
from strategy.logging_strategy import LoggingStrategy # 導入我們的具體策略

logger = logging.getLogger(__name__)

class StrategyManager:
    def __init__(self, event_queue_in: asyncio.Queue, event_queue_out: asyncio.Queue, strategy_configs: dict):
        self.event_queue_in = event_queue_in
        self.event_queue_out = event_queue_out
        self.strategy_configs = strategy_configs
        self.symbol_to_strategy_map = {}
        self._running = False
        self._processing_task = None
        
        self._init_strategies()

    def _init_strategies(self):
        # 根據設定檔，為每個 symbol 映射到一個策略實例
        for symbol, config in self.strategy_configs.items():
            strategy_type = config.get("strategy_type")
            if strategy_type == "LoggingStrategy":
                # 在真實應用中，可以為多個 symbol 共享同一個策略實例以節省記憶體
                self.symbol_to_strategy_map[symbol] = LoggingStrategy()
                logger.info(f"Mapped symbol {symbol} to LoggingStrategy.")
            else:
                logger.warning(f"Strategy type '{strategy_type}' for symbol {symbol} is not recognized.")
        
        logger.info("Strategies initialized and mapped.")

    async def _process_events(self):
        logger.info("Strategy Manager event processing started.")
        self._running = True
        while self._running:
            try:
                market_event: MarketDataEvent = await asyncio.wait_for(self.event_queue_in.get(), timeout=1.0)
                
                # 根據股票代碼查找對應的策略處理器
                strategy_handler = self.symbol_to_strategy_map.get(market_event.symbol)

                if strategy_handler:
                    # 異步調用策略的 on_data 方法
                    signals = await strategy_handler.on_data(market_event)
                    for signal in signals:
                        # 將策略產生的信號放入下一個隊列
                        await self.event_queue_out.put(signal)

                self.event_queue_in.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in Strategy Manager event processing: {e}", exc_info=True)

    def start(self):
        if not self._running:
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self):
        if self._running:
            self._running = False
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    logger.info("Strategy Manager processing task cancelled.")
            logger.info("Strategy Manager stopped.")