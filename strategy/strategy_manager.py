# strategy/strategy_manager.py
import asyncio
import logging
from collections import defaultdict
from core.event import MarketDataEvent, SignalEvent
from strategy.stateful_strategy import StatefulStrategy # 導入新策略

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
        # 按策略類型對股票進行分組
        strategies_to_load = defaultdict(list)
        for symbol, config in self.strategy_configs.items():
            strategies_to_load[config.get("strategy_type")].append(symbol)

        # 為每種類型的策略創建一個實例，並管理其對應的所有股票
        for strategy_type, symbols in strategies_to_load.items():
            if strategy_type == "StatefulStrategy":
                strategy_instance = StatefulStrategy(symbols_to_manage=symbols)
                for symbol in symbols:
                    self.symbol_to_strategy_map[symbol] = strategy_instance
            else:
                logger.warning(f"Strategy type '{strategy_type}' is not recognized.")
        
        logger.info("Strategies initialized and mapped.")

    async def _process_events(self):
        logger.info("Strategy Manager event processing started.")
        self._running = True
        while self._running:
            try:
                market_event: MarketDataEvent = await asyncio.wait_for(self.event_queue_in.get(), timeout=1.0)
                strategy_handler = self.symbol_to_strategy_map.get(market_event.symbol)

                if strategy_handler:
                    signals = await strategy_handler.on_data(market_event)
                    for signal in signals:
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