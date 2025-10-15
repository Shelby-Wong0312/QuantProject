# quant_project/core/event.py

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class Event:
    type: EventType


@dataclass
class MarketEvent(Event):
    symbol: str
    timestamp: datetime
    ohlcv_data: pd.DataFrame

    def __init__(self, symbol: str, timestamp: datetime, ohlcv_data: pd.DataFrame):
        super().__init__(EventType.MARKET)
        self.symbol = symbol
        self.timestamp = timestamp
        self.ohlcv_data = ohlcv_data


@dataclass
class SignalEvent(Event):
    symbol: str
    timestamp: datetime
    direction: str
    strategy_id: str
    quantity: float

    def __init__(
        self, symbol: str, timestamp: datetime, direction: str, strategy_id: str, quantity: float
    ):
        super().__init__(EventType.SIGNAL)
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction
        self.strategy_id = strategy_id
        self.quantity = quantity


@dataclass
class OrderEvent(Event):
    symbol: str
    timestamp: datetime
    direction: str
    quantity: float

    def __init__(self, symbol: str, timestamp: datetime, direction: str, quantity: float):
        super().__init__(EventType.ORDER)
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction
        self.quantity = quantity


@dataclass
class FillEvent(Event):
    symbol: str
    timestamp: datetime
    direction: str
    quantity: float
    fill_price: float
    commission: float

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        direction: str,
        quantity: float,
        fill_price: float,
        commission: float = 0.0,
    ):
        super().__init__(EventType.FILL)
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
