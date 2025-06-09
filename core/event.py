# core/event.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"

@dataclass
class Event:
    type: EventType

@dataclass
class MarketDataEvent(Event):
    symbol: str
    timestamp: datetime
    
    def __init__(self, symbol: str, timestamp: datetime):
        super().__init__(EventType.MARKET)
        self.symbol = symbol
        self.timestamp = timestamp

class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class SignalEvent(Event):
    symbol: str
    action: SignalAction
    quantity: float
    
    def __init__(self, symbol: str, action: SignalAction, quantity: float):
        super().__init__(EventType.SIGNAL)
        self.symbol = symbol
        self.action = action
        self.quantity = quantity

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"

@dataclass
class OrderEvent(Event):
    symbol: str
    order_type: OrderType
    action: SignalAction # BUY or SELL
    quantity: float
    
    def __init__(self, symbol: str, order_type: OrderType, action: SignalAction, quantity: float):
        super().__init__(EventType.ORDER)
        self.symbol = symbol
        self.order_type = order_type
        self.action = action
        self.quantity = quantity