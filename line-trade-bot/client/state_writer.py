"""
Helper functions for writing trading system state directly to DynamoDB.

Functions:
  - write_summary(...): update SystemState table (pk='summary')
  - append_trade_event(...): write to EventsTable (pk='EVENT') and update SystemState (pk='tradeEvents')

Usage example:

    from state_writer import write_summary, append_trade_event

    write_summary(
        system_state_table="YourSystemStateTableName",
        equity=100000.0,
        cash=60000.0,
        unrealized_pnl=800.5,
        realized_pnl=-1200.0,
        source="mybot"
    )

    append_trade_event(
        events_table="YourEventsTableName",
        system_state_table="YourSystemStateTableName",
        symbol="BTCUSDT",
        side="BUY",
        price=50000.5,
        qty=0.01,
        source="mybot",
        note="entry"
    )

Requires AWS credentials with DynamoDB write permissions.
"""

import time
import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

import boto3


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _to_decimal(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, list):
        return [_to_decimal(v) for v in value]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out[k] = _to_decimal(v)
        return out
    return value


def write_summary(
    *,
    system_state_table: str,
    equity: Optional[float] = None,
    cash: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    realized_pnl: Optional[float] = None,
    account_id: Optional[str] = None,
    region_name: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Write latest account summary to SystemState (pk='summary').

    Returns DynamoDB response from put_item.
    """
    ddb = boto3.resource("dynamodb", region_name=region_name)
    table = ddb.Table(system_state_table)
    item: Dict[str, Any] = {"pk": "summary", "updatedAt": _now_ms()}
    if equity is not None:
        item["equity"] = Decimal(str(equity))
    if cash is not None:
        item["cash"] = Decimal(str(cash))
    if unrealized_pnl is not None:
        item["unrealizedPnL"] = Decimal(str(unrealized_pnl))
    if realized_pnl is not None:
        item["realizedPnL"] = Decimal(str(realized_pnl))
    if account_id is not None:
        item["accountId"] = str(account_id)
    for k, v in extra.items():
        item[k] = _to_decimal(v)
    return table.put_item(Item=item)


def append_trade_event(
    *,
    events_table: str,
    system_state_table: str,
    symbol: str,
    side: str,
    price: Optional[float] = None,
    qty: Optional[float] = None,
    source: Optional[str] = None,
    note: Optional[str] = None,
    ts: Optional[int] = None,
    region_name: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Append a trade event: write to EventsTable and update SystemState tradeEvents cache.

    Returns dict with keys: {"put_event": ..., "update_cache": ...}
    """
    ddb = boto3.resource("dynamodb", region_name=region_name)
    events_tbl = ddb.Table(events_table)
    state_tbl = ddb.Table(system_state_table)

    now_ms = _now_ms()
    ts_str: str
    if ts is None:
        ts_str = _now_iso()
    elif isinstance(ts, (int, float)):
        # convert epoch ms to ISO8601
        sec = float(ts) / 1000.0
        dt = datetime.datetime.fromtimestamp(sec, tz=datetime.timezone.utc)
        ts_str = dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    elif isinstance(ts, str):
        ts_str = ts
    else:
        ts_str = _now_iso()

    event_item: Dict[str, Any] = {
        "ts": ts_str,
        "bucket": "ALL",
        "symbol": symbol,
        "side": side,
    }
    if price is not None:
        event_item["price"] = Decimal(str(price))
    if qty is not None:
        event_item["qty"] = _to_decimal(qty)
    if source:
        event_item["source"] = source
    if note:
        event_item["note"] = note
    for k, v in extra.items():
        event_item[k] = _to_decimal(v)

    put_resp = events_tbl.put_item(Item=event_item)

    # Update SystemState tradeEvents cache (best effort, keep last 200)
    # Using get + put to maintain a bounded list.
    cache_resp = None
    try:
        minimal_fields = {
            k: event_item.get(k)
            for k in ("symbol", "ticker", "side", "action", "qty", "quantity", "price", "source", "note", "message", "ts")
            if k in event_item
        }
        minimal_fields = _to_decimal(minimal_fields)
        cur = state_tbl.get_item(Key={"pk": "tradeEvents"}).get("Item") or {"pk": "tradeEvents", "events": []}
        events = cur.get("events") or []
        events = [minimal_fields] + events
        if len(events) > 200:
            events = events[:200]
        cur["events"] = events
        cur["updatedAt"] = now_ms
        cache_resp = state_tbl.put_item(Item=cur)
    except Exception as e:
        cache_resp = {"error": str(e)}

    return {"put_event": put_resp, "update_cache": cache_resp}
