# core/event_types.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class MarketDataEvent:
    """
    市場數據事件，封裝從數據供給器傳來的市場行情更新。
    
    """
    symbol: str
    timestamp: datetime
    event_type: str  # e.g., "AGG_MINUTE" for minute aggregate data 
    data: Dict[str, Any]  # The actual market data dictionary, e.g., OHLCV 

@dataclass
class SignalEvent:
    """
    信號事件，封裝由策略產生的交易信號。 
    """
    # --- Fields without default values FIRST ---
    symbol: str
    timestamp: datetime
    strategy_id: str
    signal_type: str  # e.g., "LONG", "SHORT", "EXIT_LONG" 
    
    # --- Fields with default values AFTER ---
    strength: float = 1.0  # Optional signal strength (e.g., for position sizing) 
    details: Optional[Dict[str, Any]] = field(default_factory=dict) # Optional details like target price 

@dataclass
class OrderEvent:
    """
    訂單事件，封裝了經過風險審核後，準備提交給券商的訂單指令。 
    """
    # --- Fields without default values FIRST ---
    symbol: str
    timestamp: datetime
    order_type: str  # e.g., "MARKET", "LIMIT" 
    side: str  # "BUY" or "SELL" 
    quantity: float

    # --- Fields with default values AFTER ---
    limit_price: Optional[float] = None # Required for LIMIT orders 
    stop_price: Optional[float] = None # Required for STOP orders 
    time_in_force: str = 'day' # e.g., "day", "gtc", "ioc" 
    client_order_id: Optional[str] = None # Optional client-side ID for tracking 
    strategy_id: Optional[str] = None # The strategy ID that generated this order 

@dataclass
class FillEvent:
    """
    成交事件，封裝了從券商處收到的關於訂單執行的成交回報。 
    """
    # --- Fields without default values FIRST ---
    symbol: str
    timestamp: datetime
    exchange: str
    quantity: float
    fill_price: float
    commission: float
    order_id: str # The broker's ID for the order 
    direction: str # "BUY" or "SELL" 

    # --- Fields with default values AFTER ---
    client_order_id: Optional[str] = None # Our internal client-side ID, if available 
    strategy_id: Optional[str] = None # The strategy ID related to this fill