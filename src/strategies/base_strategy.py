# quant_project/strategy/base_strategy.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

from core.event import SignalEvent, MarketEvent

class BaseStrategy(ABC):
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self._initialize_parameters()

    @abstractmethod
    def _initialize_parameters(self):
        pass

    @abstractmethod
    def calculate_signals(self, market_event: MarketEvent) -> Union[SignalEvent, None]:
        raise NotImplementedError("Should implement calculate_signals()")

    def _create_signal(self, direction: str, quantity: float, strategy_id: str) -> SignalEvent:
        return SignalEvent(
            symbol=self.symbol,
            timestamp=datetime.now(),
            direction=direction,
            strategy_id=strategy_id,
            quantity=quantity
        )