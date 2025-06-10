# event_types.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class MarketDataEvent:
    """
    代表市場數據更新事件(成交、報價、K線)。
    """
    event_type: str = "MARKET_DATA"
    timestamp: datetime
    symbol: str
    data_type: str # "TRADE", "QUOTE", "BAR"

    # Trade data
    price: Optional[float] = None
    volume: Optional[int] = None

    # Quote data
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    # Bar data
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    bar_volume: Optional[int] = None
    
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class SignalEvent:
    """
    代表策略產生的交易信號。
    """
    event_type: str = "SIGNAL"
    timestamp: datetime
    symbol: str
    strategy_id: str
    direction: str # "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"
    strength: float = 1.0
    order_type: str = "MARKET" # "MARKET", "LIMIT"
    limit_price: Optional[float] = None
    target_quantity: Optional[int] = None
    duration_sec: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class OrderEvent:
    """
    代表經過風險管理器批准後，準備提交給券商的訂單。
    """
    event_type: str = "ORDER"
    timestamp: datetime
    symbol: str
    order_type: str # "MARKET", "LIMIT"
    direction: str # "BUY", "SELL"
    quantity: int
    limit_price: Optional[float] = None
    client_order_id: str # 由系統生成的唯一訂單ID，用於追蹤
    strategy_id: str
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class FillEvent:
    """
    代表訂單的成交回報或狀態更新(如拒絕、取消)。
    """
    event_type: str = "FILL"
    timestamp: datetime # 來自券商的成交/狀態更新時間戳
    symbol: str
    client_order_id: str # 系統內部訂單ID
    broker_order_id: str # 券商訂單ID
    fill_id: str # 本次特定成交/執行的唯一ID (如果券商提供)
    status: str # "NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "EXPIRED"
    direction: str # "BUY", "SELL"
    
    # Fill-specific details
    fill_price: Optional[float] = None
    fill_quantity: Optional[int] = None
    
    # Cumulative details for the order
    cumulative_filled_quantity: int = 0
    average_fill_price: Optional[float] = None
    
    commission: float = 0.0
    exchange: Optional[str] = None
    remaining_quantity: int = 0
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class SystemControlEvent:
    """
    用於系統內部組件間的控制指令，如關閉指令。
    """
    event_type: str = "SYSTEM_CONTROL"
    timestamp: datetime
    command: str # e.g., "SHUTDOWN", "PAUSE_STRATEGY"
    target_component: Optional[str] = None # e.g., "StrategyManager", "ALL"
    payload: Optional[Dict[str, Any]] = field(default_factory=dict)
    