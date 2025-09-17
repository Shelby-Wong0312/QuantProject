# path: infra/state_writer.py
import os
from datetime import datetime, timezone
from typing import Optional
import boto3

AWS_REGION   = os.getenv("AWS_REGION", "ap-northeast-1")
STATE_TABLE  = os.getenv("STATE_TABLE", "SystemState")
EVENTS_TABLE = os.getenv("EVENTS_TABLE", "TradeEvents")

ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
_state  = ddb.Table(STATE_TABLE)
_events = ddb.Table(EVENTS_TABLE)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def write_summary(*, equity: float, cash: float, upnl: float, rpnl: float) -> None:
    """提供 /status 查詢用的摘要數值。"""
    _state.put_item(Item={
        "pk": "summary",
        "equity": float(equity),
        "cash": float(cash),
        "unrealized_pnl": float(upnl),
        "realized_pnl": float(rpnl),
        "updated_at": _now(),
    })

def write_positions_text(text: str) -> None:
    """提供 /positions 查詢用的多行文字。"""
    _state.put_item(Item={
        "pk": "positions",
        "text": text,
        "updated_at": _now(),
    })

def append_trade_event(*, symbol: str, side: str, quantity: float, price: float, ts: Optional[str]=None) -> None:
    """供 /last N 查詢的交易明細。Partition key 固定 'trade'。"""
    _events.put_item(Item={
        "pk": "trade",
        "ts": ts or _now(),
        "symbol": symbol,
        "side": side,
        "quantity": float(quantity),
        "price": float(price),
    })

if __name__ == "__main__":
    write_summary(equity=102345.67, cash=53421, upnl=123.45, rpnl=987.65)
    write_positions_text("XAUUSD 1@2405.0\nUS100 -2@18350.5")
    append_trade_event(symbol="US100", side="sell", quantity=2, price=18350.5)
    print("state updated.")
