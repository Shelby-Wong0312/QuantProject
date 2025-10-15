import time
from decimal import Decimal
from typing import Dict, Any
import datetime

from . import db
from .line_api import push_message


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def format_event(evt: Dict[str, Any]) -> str:
    symbol = evt.get("symbol") or evt.get("ticker") or "?"
    side = (evt.get("side") or evt.get("action") or "").upper()
    qty = evt.get("qty") or evt.get("quantity")
    price = evt.get("price")
    source = evt.get("source")
    note = evt.get("note") or evt.get("message")
    parts = ["TRADE", side, str(symbol)]
    if qty is not None:
        parts.append(str(qty))
    if price is not None:
        parts.append(f"@ {price}")
    if source:
        parts.append(f"#{source}")
    if note:
        parts.append(f"- {note}")
    return " ".join([p for p in parts if p])


def format_event_zh(evt: Dict[str, Any]) -> str:
    symbol = evt.get("symbol") or evt.get("ticker") or "?"
    side_raw = (evt.get("side") or evt.get("action") or "").upper()
    side_map = {
        "BUY": "買入",
        "SELL": "賣出",
        "LONG": "做多",
        "SHORT": "做空",
        "OPEN": "開倉",
        "CLOSE": "平倉",
        "EXIT": "平倉",
        "ADD": "加碼",
        "REDUCE": "減碼",
    }
    side = side_map.get(side_raw, side_raw or "事件")
    qty = evt.get("qty") or evt.get("quantity")
    price = evt.get("price")
    source = evt.get("source")
    note = evt.get("note") or evt.get("message")

    parts = []
    if source:
        parts.append(f"[{source}]")
    parts.append(str(symbol))
    if side:
        parts.append(str(side))
    if qty is not None:
        parts.append(str(qty))
    if price is not None:
        parts.append(f"@ {price}")
    if note:
        parts.append(f"｜{note}")
    return "交易事件：" + " ".join([p for p in parts if p])


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    item: Dict[str, Any] = {"ts": _now_iso(), "bucket": "ALL"}
    for k in (
        "symbol",
        "ticker",
        "side",
        "action",
        "qty",
        "quantity",
        "price",
        "source",
        "note",
        "message",
    ):
        if k in payload:
            v = payload[k]
            if isinstance(v, float):
                v = Decimal(str(v))
            item[k] = v
    return item


def process_payload(
    payload: Dict[str, Any], *, formatter: str = "en"
) -> Dict[str, Any]:
    item = normalize_payload(payload)
    db.put_event(item)
    try:
        db.append_trade_event_recent(item)
    except Exception:
        pass
    if formatter == "zh":
        text = format_event_zh(item)
    else:
        text = format_event(item)
    subs = db.list_subscribers()
    delivered = 0
    for s in subs:
        rid = s.get("recipientId")
        if not rid:
            continue
        # whitelist check (if configured)
        try:
            if not db.is_target_whitelisted(rid):
                continue
        except Exception:
            pass
        res = push_message(rid, text)
        if not res.get("error"):
            delivered += 1
    return {"ok": True, "delivered": delivered, "subscribers": len(subs), "item": item}


def format_status_zh(status: Dict[str, Any]) -> str:
    eq = status.get("equity")
    cash = status.get("cash")
    upnl = (
        status.get("unrealizedPnL")
        or status.get("unrealized_pnl")
        or status.get("upnl")
    )
    rpnl = status.get("realizedPnL") or status.get("realized_pnl") or status.get("rpnl")
    acct = status.get("accountId") or status.get("account") or "default"
    parts = [f"帳戶：{acct}"]
    if eq is not None:
        parts.append(f"Equity：{eq}")
    if cash is not None:
        parts.append(f"Cash：{cash}")
    if upnl is not None:
        parts.append(f"未實現PnL：{upnl}")
    if rpnl is not None:
        parts.append(f"已實現PnL：{rpnl}")
    return "狀態摘要\n" + "\n".join(parts)


def format_positions_zh(snapshot: Dict[str, Any]) -> str:
    acct = snapshot.get("accountId") or snapshot.get("account") or "default"
    positions = snapshot.get("positions") or []
    if not positions:
        return f"持倉摘要\n帳戶：{acct}\n目前無持倉"
    side_map = {
        "LONG": "多",
        "SHORT": "空",
        "BUY": "多",
        "SELL": "空",
    }
    lines = []
    for p in positions:
        symbol = p.get("symbol") or p.get("ticker") or "?"
        side_raw = (p.get("side") or p.get("direction") or "").upper()
        side = side_map.get(side_raw, side_raw or "-")
        qty = p.get("qty") or p.get("size") or p.get("quantity")
        avg = p.get("avgPrice") or p.get("entryPrice") or p.get("avg") or p.get("price")
        upnl = p.get("unrealizedPnL") or p.get("upnl") or p.get("pnl")
        parts = [str(symbol), str(side)]
        if qty is not None:
            parts.append(str(qty))
        if avg is not None:
            parts.append(f"@ {avg}")
        detail = " ".join(parts)
        if upnl is not None:
            detail += f"｜未實現PnL：{upnl}"
        lines.append(f"- {detail}")
    return "持倉摘要\n帳戶：" + acct + "\n" + "\n".join(lines)
