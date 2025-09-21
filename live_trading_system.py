# live_trading_system.py
from __future__ import annotations
import logging, os, threading, time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from infra.publish_to_sns import publish_trade_event
from infra.state_writer import write_summary, write_positions_text, append_trade_event

logging.basicConfig(level=os.getenv("LTS_LOG_LEVEL","INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("live_trading_system")

@dataclass
class Order:
    symbol: str; side: str; quantity: float
    price: Optional[float] = None; deal_id: Optional[str]=None
    extra: Optional[Dict[str, Any]] = None

@dataclass
class Fill:
    symbol: str; side: str; quantity: float; price: float
    pnl: Optional[float] = None; deal_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

@dataclass
class Position:
    symbol: str; quantity: float; avg_price: float

class BrokerAdapter:
    @property
    def equity(self) -> float: raise NotImplementedError
    @property
    def cash(self) -> float:   raise NotImplementedError
    @property
    def upnl(self) -> float:   raise NotImplementedError
    @property
    def rpnl(self) -> float:   raise NotImplementedError
    def positions(self) -> Iterable[Position]: raise NotImplementedError

def _fmt_num(v: Optional[float], nd: int = 4) -> str:
    try:
        x = round(float(v), nd)
        if abs(x) < 10 ** (-nd): x = 0.0
        return f"{x:.{nd}f}"
    except Exception:
        return "0"

def positions_as_text(positions: Iterable[Position]) -> str:
    lines=[]
    for p in positions:
        sign = "+" if p.quantity>0 else ""
        lines.append(f"{p.symbol} {sign}{_fmt_num(p.quantity,4)} @ {_fmt_num(p.avg_price,4)}")
    return "\n".join(lines) if lines else "No positions"

class LiveTradingSystem:
    def __init__(self, broker: BrokerAdapter, snapshot_enabled: bool = True):
        self.broker = broker
        self._snapshot_enabled = snapshot_enabled
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        log.info("LiveTradingSystem ready (snapshot_enabled=%s)", snapshot_enabled)

    def on_order_submitted(self, order: Order) -> None:
        side = (order.side or "").lower()
        payload = {
            "symbol": order.symbol, "side": side if side in ("buy","sell") else "buy",
            "quantity": float(order.quantity),
            "price": float(order.price) if order.price is not None else 0.0,
            "status": "submitted", "deal_id": order.deal_id, "dealId": order.deal_id,
            "extra": order.extra or {},
        }
        try:
            publish_trade_event(**payload)
            log.info("submitted -> SNS: %s", payload)
        except Exception as e:
            log.warning("publish submitted failed: %s", e)

    def on_order_filled(self, fill: Fill) -> None:
        side = (fill.side or "").lower()
        payload = {
            "symbol": fill.symbol, "side": side if side in ("buy","sell") else "buy",
            "quantity": float(fill.quantity), "price": float(fill.price),
            "status": "filled", "deal_id": fill.deal_id, "dealId": fill.deal_id,
            "pnl": float(fill.pnl) if fill.pnl is not None else None,
            "extra": fill.extra or {},
        }
        try:
            publish_trade_event(**payload)
            log.info("filled -> SNS: %s", payload)
        except Exception as e:
            log.warning("publish filled failed: %s", e)
        try:
            # /last N：帶上 pnl（BUY 無 pnl 會省略）
            append_trade_event(symbol=fill.symbol, side=payload["side"],
                               quantity=float(fill.quantity), price=float(fill.price),
                               pnl=(float(fill.pnl) if fill.pnl is not None else None))
        except Exception as e:
            log.warning("append_trade_event failed: %s", e)

    def sync_state_snapshot(self) -> None:
        try:
            equity=float(getattr(self.broker,"equity")); cash=float(getattr(self.broker,"cash"))
            upnl=float(getattr(self.broker,"upnl"));     rpnl=float(getattr(self.broker,"rpnl"))
            write_summary(equity=equity, cash=cash, upnl=upnl, rpnl=rpnl)
        except Exception as e:
            log.warning("write_summary failed: %s", e)
        try:
            text = positions_as_text(self.broker.positions())
            write_positions_text(text)
        except Exception as e:
            log.warning("write_positions_text failed: %s", e)

    def start_state_sync(self, period_sec: int = 60) -> None:
        if getattr(self, "_th", None) and self._th.is_alive(): return
        self._stop.clear()
        def _loop():
            log.info("state sync loop started (period=%ss)", period_sec)
            while not self._stop.is_set():
                try:
                    if self._snapshot_enabled: self.sync_state_snapshot()
                except Exception as e:
                    log.warning("snapshot loop error: %s", e)
                finally:
                    for _ in range(max(1,int(period_sec))):
                        if self._stop.is_set(): break
                        time.sleep(1)
            log.info("state sync loop stopped")
        self._th = threading.Thread(target=_loop, name="lts-sync", daemon=True); self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if getattr(self,"_th",None) and self._th.is_alive(): self._th.join(timeout=5)
