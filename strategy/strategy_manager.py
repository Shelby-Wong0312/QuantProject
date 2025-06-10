# strategy/strategy_manager.py

import asyncio
import logging
from collections import deque
from typing import List, Dict, Any, Type

from core.event_types import MarketDataEvent, SignalEvent
from core import config
from core import utils # <--- 修正引用方式

logger = logging.getLogger(__name__)

class BaseStrategy:
    # ... (類別內部程式碼不變，僅修正引用後的函式調用) ...
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any], signal_queue: asyncio.Queue):
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.params = params
        self.signal_queue = signal_queue
        self.symbol_state: Dict[str, Dict[str, Any]] = {sym: {} for sym in symbols}
        logger.info(f"Strategy {self.strategy_id} initialized for symbols {self.symbols} with params {self.params}")

    async def on_market_data(self, event: MarketDataEvent):
        raise NotImplementedError("Each strategy must implement on_market_data")

    async def _generate_signal(self, symbol: str, direction: str, strength: float = 1.0,
                              order_type: str = "MARKET", limit_price: float = None,
                              target_quantity: int = None):
        signal = SignalEvent(
            # 使用 utils.get_current_timestamp()
            timestamp=utils.get_current_timestamp(),
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            strength=strength,
            order_type=order_type,
            limit_price=limit_price,
            target_quantity=target_quantity
        )
        await self.signal_queue.put(signal)
        logger.info(f"Strategy {self.strategy_id} generated signal: {signal}")
# ... (MovingAverageCrossoverStrategy 和 StrategyManager 的其餘部分不變) ...
class MovingAverageCrossoverStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any], signal_queue: asyncio.Queue):
        super().__init__(strategy_id, symbols, params, signal_queue)
        self.short_window = self.params.get("short_window", 10)
        self.long_window = self.params.get("long_window", 20)
        for symbol in self.symbols:
            self.symbol_state[symbol]['prices'] = deque(maxlen=self.long_window)
            self.symbol_state[symbol]['short_sma'] = None
            self.symbol_state[symbol]['long_sma'] = None
            self.symbol_state[symbol]['position'] = 0

    async def on_market_data(self, event: MarketDataEvent):
        if event.symbol not in self.symbols or event.data_type != "TRADE" or event.price is None:
            return
        state = self.symbol_state[event.symbol]
        prices = state['prices']
        prices.append(event.price)
        if len(prices) < self.long_window:
            return
        short_sma_val = sum(list(prices)[-self.short_window:]) / self.short_window
        long_sma_val = sum(prices) / self.long_window
        prev_short_sma = state.get('short_sma')
        prev_long_sma = state.get('long_sma')
        state['short_sma'] = short_sma_val
        state['long_sma'] = long_sma_val
        logger.debug(f"{self.strategy_id} - {event.symbol}: ShortSMA={short_sma_val:.2f}, LongSMA={long_sma_val:.2f}, Position={state['position']}")
        if prev_short_sma is not None and prev_long_sma is not None:
            if prev_short_sma <= prev_long_sma and short_sma_val > long_sma_val and state['position'] <= 0:
                logger.info(f"{self.strategy_id} - {event.symbol}: BUY signal triggered.")
                state['position'] = 1
                await self._generate_signal(
                    symbol=event.symbol, direction="BUY",
                    target_quantity=config.MAX_ORDER_QUANTITY_PER_TRADE // 2
                )
            elif prev_short_sma >= prev_long_sma and short_sma_val < long_sma_val and state['position'] >= 0:
                logger.info(f"{self.strategy_id} - {event.symbol}: SELL signal triggered.")
                state['position'] = -1
                await self._generate_signal(
                    symbol=event.symbol, direction="SELL",
                    target_quantity=config.MAX_ORDER_QUANTITY_PER_TRADE // 2
                )

STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {
    "MovingAverageCrossoverStrategy": MovingAverageCrossoverStrategy,
}

class StrategyManager:
    def __init__(self, 
                 market_data_queue: asyncio.Queue,
                 signal_queue: asyncio.Queue,
                 strategies_config: List[Dict] = config.STRATEGIES_CONFIG):
        self.market_data_queue = market_data_queue
        self.signal_queue = signal_queue
        self.strategies_config = strategies_config
        self.strategies: List[BaseStrategy] = []
        self._running = False
        
    def _load_strategies(self):
        self.strategies = []
        for strat_conf in self.strategies_config:
            class_name = strat_conf.get("class_name")
            strategy_id = strat_conf.get("id")
            if not class_name or not strategy_id:
                logger.error(f"Strategy config missing class_name or id: {strat_conf}")
                continue
            StrategyClass = STRATEGY_CLASSES.get(class_name)
            if StrategyClass:
                try:
                    symbols = strat_conf.get("symbols", [])
                    params = strat_conf.get("params", {})
                    strategy_instance = StrategyClass(strategy_id, symbols, params, self.signal_queue)
                    self.strategies.append(strategy_instance)
                    logger.info(f"Loaded strategy: {strategy_id} of type {class_name}")
                except Exception as e:
                    logger.error(f"Error instantiating strategy {strategy_id} ({class_name}): {e}", exc_info=True)
            else:
                logger.error(f"Strategy class {class_name} not found for strategy {strategy_id}.")
        if not self.strategies:
            logger.warning("No strategies were loaded.")

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        self._load_strategies()
        logger.info(f"StrategyManager starting with {len(self.strategies)} strategies...")
        while self._running and not shutdown_event.is_set():
            try:
                market_event: MarketDataEvent = await asyncio.wait_for(
                    self.market_data_queue.get(), timeout=1.0
                )
                for strategy in self.strategies:
                    if market_event.symbol in strategy.symbols:
                        try:
                            await strategy.on_market_data(market_event)
                        except Exception as e:
                            logger.error(f"Error in strategy {strategy.strategy_id} processing event {market_event}: {e}", exc_info=True)
                self.market_data_queue.task_done()
            except asyncio.TimeoutError:
                if shutdown_event.is_set():
                    logger.info("StrategyManager: Shutdown signaled.")
                    break
                continue
            except Exception as e:
                logger.error(f"StrategyManager: An error occurred in run loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        await self.stop()
        logger.info("StrategyManager stopped.")
        
    async def stop(self):
        self._running = False
        logger.info("StrategyManager stopping...")
        logger.info(f"StrategyManager: {len(self.strategies)} strategies processed for stop.")