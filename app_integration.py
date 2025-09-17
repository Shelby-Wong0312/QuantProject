# path: app_integration.py
"""
把你的交易引擎接到 LINE 回報／DynamoDB 快照的最薄一層。
使用方式：
    from app_integration import init_lts, start_background_sync, notify_submitted, notify_filled, shutdown

    lts = init_lts(engine)             # 程式啟動時
    start_background_sync()            # 可選，預設 60s 寫一次 /status 和 /positions
    notify_submitted(symbol, side, qty, price, deal_id)   # 下單被接受時呼叫
    notify_filled(symbol, side, qty, price, pnl, deal_id) # 成交/回報時呼叫
    shutdown()                         # 程式結束時
"""
from __future__ import annotations
import os, logging
from typing import Optional, Iterable
from dataclasses import dataclass

from live_trading_system import (
    LiveTradingSystem, Order, Fill, Position, BrokerAdapter
)

log = logging.getLogger("app_integration")
_lts: Optional[LiveTradingSystem] = None

# --- 你現有引擎需要包一層 BrokerAdapter，提供 equity/cash/upnl/rpnl/positions() ---
@dataclass
class _Pos:
    symbol: str
    qty: float
    avg_price: float

class _MyBrokerAdapter(BrokerAdapter):
    def __init__(self, engine):
        self.engine = engine

    @property
    def equity(self) -> float: return float(getattr(self.engine, "equity"))
    @property
    def cash(self)   -> float: return float(getattr(self.engine, "cash"))
    @property
    def upnl(self)   -> float: return float(getattr(self.engine, "upnl"))
    @property
    def rpnl(self)   -> float: return float(getattr(self.engine, "rpnl"))

    def positions(self) -> Iterable[Position]:
        """
        依你的引擎狀態把部位轉成 Position；請把下面兩行替換成你的資料來源。
        例：self.engine.open_positions 內含欄位 symbol/qty/avg_price
        """
        for p in getattr(self.engine, "open_positions", []):
            yield Position(symbol=p.symbol, quantity=float(p.qty), avg_price=float(p.avg_price))

# ---- 對外 API（你程式只需要呼叫這幾個函式） ----
def init_lts(engine) -> LiveTradingSystem:
    """在程式啟動時呼叫一次。"""
    global _lts
    adapter = _MyBrokerAdapter(engine)
    _lts = LiveTradingSystem(adapter, snapshot_enabled=True)
    log.info("LTS initialized.")
    return _lts

def start_background_sync(period_sec: int = 60) -> None:
    """啟動背景同步（寫 /status 與 /positions 到 DynamoDB）。"""
    if _lts is None: raise RuntimeError("init_lts() not called")
    _lts.start_state_sync(period_sec=period_sec)

def shutdown() -> None:
    """程式結束時呼叫，乾淨停掉背景執行緒。"""
    if _lts: _lts.stop()

def notify_submitted(symbol: str, side: str, qty: float, price: float, deal_id: Optional[str]=None) -> None:
    """下單被接受/送出成功時呼叫。"""
    if _lts is None: raise RuntimeError("init_lts() not called")
    _lts.on_order_submitted(Order(symbol=symbol, side=side.lower(), quantity=float(qty), price=float(price), deal_id=deal_id))

def notify_filled(symbol: str, side: str, qty: float, price: float, pnl: Optional[float]=None, deal_id: Optional[str]=None) -> None:
    """成交/回報時呼叫。"""
    if _lts is None: raise RuntimeError("init_lts() not called")
    _lts.on_order_filled(Fill(symbol=symbol, side=side.lower(), quantity=float(qty), price=float(price), pnl=(float(pnl) if pnl is not None else None), deal_id=deal_id))
