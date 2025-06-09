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
    # ...其他市場數據欄位，例如 price, volume 等
    
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