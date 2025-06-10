# strategy/strategy_manager.py

import asyncio
import logging
from collections import deque
from typing import List, Dict, Any, Type

from core.event_types import MarketDataEvent, SignalEvent
from core import config
from core.utils import get_current_timestamp

logger = logging.getLogger(__name__)


class BaseStrategy:
    """
    所有策略類的抽象基類，定義了策略所需的基本接口。
    """
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any], signal_queue: asyncio.Queue):
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.params = params
        self.signal_queue = signal_queue
        self.symbol_state: Dict[str, Dict[str, Any]] = {sym: {} for sym in symbols}
        logger.info(f"Strategy {self.strategy_id} initialized for symbols {self.symbols} with params {self.params}")

    async def on_market_data(self, event: MarketDataEvent):
        """
        處理傳入的市場數據。每個具體的策略子類都必須實現此方法。
        """
        raise NotImplementedError("Each strategy must implement on_market_data")

    async def _generate_signal(self, symbol: str, direction: str, strength: float = 1.0,
                              order_type: str = "MARKET", limit_price: float = None,
                              target_quantity: int = None):
        """
        輔助方法，用於創建並發送一個 SignalEvent。
        """
        signal = SignalEvent(
            timestamp=get_current_timestamp(),
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


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    一個簡單的移動平均線交叉策略範例。
    """
    def __init__(self, strategy_id: str, symbols: List[str], params: Dict[str, Any], signal_queue: asyncio.Queue):
        super().__init__(strategy_id, symbols, params, signal_queue)
        self.short_window = self.params.get("short_window", 10)
        self.long_window = self.params.get("long_window", 20)
        
        for symbol in self.symbols:
            self.symbol_state[symbol]['prices'] = deque(maxlen=self.long_window)
            self.symbol_state[symbol]['short_sma'] = None
            self.symbol_state[symbol]['long_sma'] = None
            self.symbol_state[symbol]['position'] = 0  # 0: flat, 1: long, -1: short

    async def on_market_data(self, event: MarketDataEvent):
        if event.symbol not in self.symbols or event.data_type != "TRADE" or event.price is None:
            return

        state = self.symbol_state[event.symbol]
        prices = state['prices']
        prices.append(event.price)

        if len(prices) < self.long_window:
            return  # 數據不足

        # 計算移動平均線
        short_sma_val = sum(list(prices)[-self.short_window:]) / self.short_window
        long_sma_val = sum(prices) / self.long_window # deque 已限制長度

        prev_short_sma = state.get('short_sma')
        prev_long_sma = state.get('long_sma')

        state['short_sma'] = short_sma_val
        state['long_sma'] = long_sma_val
        
        logger.debug(f"{self.strategy_id} - {event.symbol}: ShortSMA={short_sma_val:.2f}, LongSMA={long_sma_val:.2f}, Position={state['position']}")

        # 交叉邏輯
        if prev_short_sma is not None and prev_long_sma is not None:
            # 黃金交叉: 短期線上穿長期線 -> 買入信號
            if prev_short_sma <= prev_long_sma and short_sma_val > long_sma_val and state['position'] <= 0:
                logger.info(f"{self.strategy_id} - {event.symbol}: BUY signal triggered.")
                state['position'] = 1
                await self._generate_signal(
                    symbol=event.symbol,
                    direction="BUY",
                    target_quantity=config.MAX_ORDER_QUANTITY_PER_TRADE // 2 # 範例數量
                )
            # 死亡交叉: 短期線下穿長期線 -> 賣出信號
            elif prev_short_sma >= prev_long_sma and short_sma_val < long_sma_val and state['position'] >= 0:
                logger.info(f"{self.strategy_id} - {event.symbol}: SELL signal triggered.")
                state['position'] = -1
                await self._generate_signal(
                    symbol=event.symbol,
                    direction="SELL",
                    target_quantity=config.MAX_ORDER_QUANTITY_PER_TRADE // 2 # 範例數量
                )


# 策略註冊表，用於動態載入策略
STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {
    "MovingAverageCrossoverStrategy": MovingAverageCrossoverStrategy,
    # 在此處加入其他策略類
}


class StrategyManager:
    """
    管理所有活躍的交易策略。
    """
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
        """從設定檔載入並初始化策略。"""
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
                    self.market_data_queue.get(),
                    timeout=1.0 # 設置超時以允許檢查關閉信號
                )
                
                # 將事件路由到所有相關的策略
                for strategy in self.strategies:
                    if market_event.symbol in strategy.symbols:
                        try:
                            await strategy.on_market_data(market_event)
                        except Exception as e:
                            logger.error(f"Error in strategy {strategy.strategy_id} processing event {market_event}: {e}", exc_info=True)
                
                self.market_data_queue.task_done()

            except asyncio.TimeoutError:
                # 佇列為空時，這是預期行為，讓我們可以檢查關閉信號
                if shutdown_event.is_set():
                    logger.info("StrategyManager: Shutdown signaled.")
                    break
                continue
            except Exception as e:
                logger.error(f"StrategyManager: An error occurred in run loop: {e}", exc_info=True)
                # 可選擇在此處短暫休眠以防止快速錯誤循環
                await asyncio.sleep(1)

        await self.stop()
        logger.info("StrategyManager stopped.")
        
    async def stop(self):
        self._running = False
        logger.info("StrategyManager stopping...")
        # 可在此處加入策略清理iffs
        logger.info(f"StrategyManager: {len(self.strategies)} strategies processed for stop.")