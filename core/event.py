# 檔案位置: core/event.py

from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import uuid

# --- 基礎事件 ---
@dataclass
class BaseEvent:
    """所有事件的基類，自帶唯一的事件ID和時間戳。"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

# --- 市場事件 ---
@dataclass
class MarketDataEvent(BaseEvent):
    """代表新的市場數據（通常是一根K棒）已到達。"""
    # --- 以下是修正的部分 ---
    symbol: str = None
    ohlcv_data: pd.DataFrame = None

# --- 策略與訂單事件 ---
@dataclass
class SignalEvent(BaseEvent):
    """代表策略已產生一個交易訊號。"""
    # --- 以下是修正的部分 ---
    strategy_id: str = None
    symbol: str = None
    action: str = None
    # --- 修正結束 ---
    quantity: float = None
    price: float = None
    sl_price: float = None
    tp_price: float = None
    order_type: str = 'MARKET'
    comment: str = ''
    correlation_id: str = None

@dataclass
class OrderEvent(BaseEvent):
    """代表一個交易訂單已被創建並發送至券商。"""
    # --- 以下是修正的部分 ---
    symbol: str = None
    action: str = None
    quantity: float = None
    order_type: str = 'MARKET'
    # --- 修正結束 ---
    internal_order_id: str = field(default_factory=lambda: f"ORD_{uuid.uuid4()}")
    status: str = 'PENDING_SUBMIT'
    correlation_id: str = None

@dataclass
class FillEvent(BaseEvent):
    """代表一個訂單已被部分或全部成交。"""
    # --- 以下是修正的部分 ---
    symbol: str = None
    action: str = None
    fill_quantity: float = None
    fill_price: float = None
    internal_order_id: str = None
    # --- 修正結束 ---
    commission: float = 0.0
    broker_trade_id: str = ''
    correlation_id: str = None

# --- 系統級事件 ---
@dataclass
class SystemEvent(BaseEvent):
    """代表系統級別的事件，如啟動、關閉、錯誤等。"""
    # --- 以下是修正的部分 ---
    event_type: str = None
    message: str = None
    # --- 修正結束 ---
    severity: str = 'INFO'
    