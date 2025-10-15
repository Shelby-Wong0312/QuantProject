# path: live_trading_system.py
"""
把你的交易引擎接上：
  - 交易事件 -> SNS -> (line-push) -> LINE 群
  - 摘要/持倉/交易 -> DynamoDB -> LINE 指令 (/status /positions /last N)

需要環境變數（在交易主機/容器）：
  TRADE_EVENTS_TOPIC_ARN=arn:aws:sns:ap-northeast-1:<acct>:trade-events
  STATE_TABLE=SystemState
  EVENTS_TABLE=TradeEvents
  AWS_REGION=ap-northeast-1  # 可省略，預設即此
以及執行身分最小 IAM：
  sns:Publish -> 上述 topic
  dynamodb:PutItem -> SystemState、TradeEvents
"""
from __future__ import annotations

import logging, os, threading, time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from infra.publish_to_sns import publish_trade_event
from infra.state_writer import write_summary, write_positions_text, append_trade_event

# ---- logging ----
logging.basicConfig(
    level=os.getenv("LTS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("live_trading_system")


# ---- domain models ----
@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    price: Optional[float] = None
    deal_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Fill:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    pnl: Optional[float] = None
    deal_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    symbol: str
    quantity: float  # 正負代表多/空
    avg_price: float


class BrokerAdapter:
    """你的引擎需提供：equity/cash/upnl/rpnl 屬性與 positions() 列舉 Position。"""

    @property
    def equity(self) -> float:
        raise NotImplementedError

    @property
    def cash(self) -> float:
        raise NotImplementedError

    @property
    def upnl(self) -> float:
        raise NotImplementedError

    @property
    def rpnl(self) -> float:
        raise NotImplementedError

    def positions(self) -> Iterable[Position]:
        raise NotImplementedError


# ---- helpers ----
def _fmt_num(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "0"
    x = round(float(v), nd)
    if abs(x) < 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}"


def positions_as_text(positions: Iterable[Position]) -> str:
    lines: List[str] = []
    for p in positions:
        sign = "+" if p.quantity > 0 else ""
        lines.append(f"{p.symbol} {sign}{_fmt_num(p.quantity,4)} @ { _fmt_num(p.avg_price,4) }")
    return "\n".join(lines) if lines else "No positions"


# ---- main wiring ----
class LiveTradingSystem:
    def __init__(self, broker: BrokerAdapter, snapshot_enabled: bool = True):
        self.broker = broker
        self._snapshot_enabled = snapshot_enabled
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        log.info("LiveTradingSystem ready (snapshot_enabled=%s)", snapshot_enabled)

    # ----- 事件掛點：請在你的程式對應位置呼叫 -----
    def on_order_submitted(self, order: Order) -> None:
        side = (order.side or "").lower()
        payload = {
            "symbol": order.symbol,
            "side": side if side in ("buy", "sell") else "buy",
            "quantity": float(order.quantity),
            "price": float(order.price) if order.price is not None else 0.0,
            "status": "submitted",
            "dealId": order.deal_id,
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
            "symbol": fill.symbol,
            "side": side if side in ("buy", "sell") else "buy",
            "quantity": float(fill.quantity),
            "price": float(fill.price),
            "status": "filled",
            "dealId": fill.deal_id,
            "pnl": float(fill.pnl) if fill.pnl is not None else None,
            "extra": fill.extra or {},
        }
        try:
            publish_trade_event(**payload)
            log.info("filled -> SNS: %s", payload)
        except Exception as e:
            log.warning("publish filled failed: %s", e)
        # /last N 查詢用
        try:
            append_trade_event(
                symbol=fill.symbol,
                side=payload["side"],
                quantity=float(fill.quantity),
                price=float(fill.price),
            )
        except Exception as e:
            log.warning("append_trade_event failed: %s", e)

    def on_order_rejected(self, order: Order, reason: str) -> None:
        side = (order.side or "").lower()
        payload = {
            "symbol": order.symbol,
            "side": side if side in ("buy", "sell") else "buy",
            "quantity": float(order.quantity),
            "price": float(order.price) if order.price is not None else 0.0,
            "status": "rejected",
            "dealId": order.deal_id,
            "extra": {"reason": reason, **(order.extra or {})},
        }
        try:
            publish_trade_event(**payload)
            log.info("rejected -> SNS: %s", payload)
        except Exception as e:
            log.warning("publish rejected failed: %s", e)

    # ----- 週期性快照：供 /status、/positions -----
    def sync_state_snapshot(self) -> None:
        try:
            equity = float(getattr(self.broker, "equity"))
            cash = float(getattr(self.broker, "cash"))
            upnl = float(getattr(self.broker, "upnl"))
            rpnl = float(getattr(self.broker, "rpnl"))
        except Exception as e:
            log.warning("read broker summary failed: %s", e)
            return
        try:
            write_summary(equity=equity, cash=cash, upnl=upnl, rpnl=rpnl)
        except Exception as e:
            log.warning("write_summary failed: %s", e)
        try:
            text = positions_as_text(self.broker.positions())
            write_positions_text(text)
        except Exception as e:
            log.warning("write_positions_text failed: %s", e)
        log.debug("snapshot synced.")

    def start_state_sync(self, period_sec: int = 60) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def _loop():
            log.info("state sync loop started (period=%ss)", period_sec)
            while not self._stop.is_set():
                try:
                    if self._snapshot_enabled:
                        self.sync_state_snapshot()
                except Exception as e:
                    log.warning("snapshot loop error: %s", e)
                finally:
                    for _ in range(max(1, int(period_sec))):
                        if self._stop.is_set():
                            break
                        time.sleep(1)
            log.info("state sync loop stopped")

        self._th = threading.Thread(target=_loop, name="lts-sync", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th and self._th.is_alive():
            self._th.join(timeout=5)


# ---- 可選：本檔直接跑做快速驗收 ----
if __name__ == "__main__":

    class _DemoBroker(BrokerAdapter):
        def __init__(self):
            self._equity = 100000.0
            self._cash = 60000.0
            self._upnl = 12.34
            self._rpnl = 345.67
            self._pos = [Position("US100", -2, 18350.5), Position("XAUUSD", 1, 2405.0)]

        @property
        def equity(self):
            return self._equity

        @property
        def cash(self):
            return self._cash

        @property
        def upnl(self):
            return self._upnl

        @property
        def rpnl(self):
            return self._rpnl

        def positions(self):
            return list(self._pos)

    demo = _DemoBroker()
    lts = LiveTradingSystem(demo, snapshot_enabled=True)
    lts.sync_state_snapshot()  # 寫一次 /status、/positions
    lts.on_order_submitted(Order("XAUUSD", "buy", 1, 2405.0, "SUB123"))
    lts.on_order_filled(Fill("XAUUSD", "buy", 1, 2405.2, 12.3, "FILL123"))
    lts.start_state_sync(period_sec=30)
    time.sleep(60)
    lts.stop()
