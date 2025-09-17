import os
import time
import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any

import boto3
from boto3.dynamodb.conditions import Key


_dynamodb = boto3.resource("dynamodb")
_subs_table = _dynamodb.Table(os.environ.get("SUBSCRIBERS_TABLE", ""))
_events_table = _dynamodb.Table(os.environ.get("EVENTS_TABLE", ""))
_line_groups_table_name = (
    os.environ.get("GROUPS_TABLE")
    or os.environ.get("LINE_GROUPS_TABLE")
    or os.environ.get("GROUP_WHITELIST_TABLE", "")
)
_line_groups_table = _dynamodb.Table(_line_groups_table_name) if _line_groups_table_name else None
_system_state_table_name = os.environ.get("SYSTEM_STATE_TABLE") or os.environ.get("STATE_TABLE", "")
_system_state_table = _dynamodb.Table(_system_state_table_name) if _system_state_table_name else None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def register_subscriber(recipient_id: str, recipient_type: str) -> None:
    _subs_table.put_item(
        Item={
            "recipientId": recipient_id,
            "type": recipient_type,
            "createdAt": _now_ms(),
        }
    )


def unregister_subscriber(recipient_id: str) -> None:
    _subs_table.delete_item(Key={"recipientId": recipient_id})


def get_subscriber(recipient_id: str) -> Optional[Dict[str, Any]]:
    resp = _subs_table.get_item(Key={"recipientId": recipient_id})
    return resp.get("Item")


def list_subscribers() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    start_key = None
    while True:
        if start_key:
            resp = _subs_table.scan(ExclusiveStartKey=start_key)
        else:
            resp = _subs_table.scan()
        items.extend(resp.get("Items", []))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break
    return items


def put_event(item: Dict[str, Any]) -> None:
    if "ts" not in item:
        item["ts"] = _now_iso()
    if "bucket" not in item:
        item["bucket"] = "ALL"
    for k, v in list(item.items()):
        if isinstance(v, float):
            item[k] = Decimal(str(v))
    _events_table.put_item(Item=item)


def get_last_events(limit: int = 5) -> List[Dict[str, Any]]:
    resp = _events_table.query(
        IndexName="recent-index",
        KeyConditionExpression=Key("bucket").eq("ALL"),
        ScanIndexForward=False,
        Limit=max(1, min(limit, 50)),
    )
    return resp.get("Items", [])


def put_status(account_id: str, status: Dict[str, Any]) -> None:
    # Write to SystemState summary (compat shim)
    s = {
        "equity": status.get("equity"),
        "cash": status.get("cash"),
        "unrealizedPnL": status.get("unrealizedPnL") or status.get("unrealized_pnl") or status.get("upnl"),
        "realizedPnL": status.get("realizedPnL") or status.get("realized_pnl") or status.get("rpnl"),
        "accountId": account_id,
    }
    put_system_summary(s)


def get_latest_status(account_id: str) -> Optional[Dict[str, Any]]:
    return get_system_summary()


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


def put_positions(account_id: str, snapshot: Dict[str, Any]) -> None:
    put_system_positions({"accountId": account_id, **(snapshot or {})})


def get_latest_positions(account_id: str) -> Optional[Dict[str, Any]]:
    return get_system_positions()


def put_system_summary(status: Dict[str, Any]) -> None:
    if not _system_state_table:
        return
    item: Dict[str, Any] = {"pk": "summary", "updatedAt": _now_ms()}
    for k, v in status.items():
        if isinstance(v, float):
            item[k] = Decimal(str(v))
        else:
            item[k] = v
    _system_state_table.put_item(Item=item)


def get_system_summary() -> Optional[Dict[str, Any]]:
    if not _system_state_table:
        return None
    resp = _system_state_table.get_item(Key={"pk": "summary"})
    return resp.get("Item")


def put_system_positions(snapshot: Dict[str, Any]) -> None:
    if not _system_state_table:
        return
    item: Dict[str, Any] = {"pk": "positions", "updatedAt": _now_ms()}
    item.update(_to_decimal(snapshot) if isinstance(snapshot, dict) else {"raw": snapshot})
    _system_state_table.put_item(Item=item)


def get_system_positions() -> Optional[Dict[str, Any]]:
    if not _system_state_table:
        return None
    resp = _system_state_table.get_item(Key={"pk": "positions"})
    return resp.get("Item")


def append_trade_event_recent(event_item: Dict[str, Any], *, max_items: int = 200) -> None:
    if not _system_state_table:
        return
    # project minimal fields to store
    fields = ("symbol", "ticker", "side", "action", "qty", "quantity", "price", "source", "note", "message", "ts")
    e = {k: event_item.get(k) for k in fields if k in event_item}
    e = _to_decimal(e)
    try:
        resp = _system_state_table.get_item(Key={"pk": "tradeEvents"})
        item = resp.get("Item") or {"pk": "tradeEvents", "events": []}
        events = item.get("events") or []
        # prepend and trim
        events = [e] + events
        if len(events) > max_items:
            events = events[:max_items]
        item["events"] = events
        item["updatedAt"] = _now_ms()
        _system_state_table.put_item(Item=item)
    except Exception:
        # best-effort cache
        pass


def get_trade_events_recent(limit: int = 5) -> List[Dict[str, Any]]:
    if not _system_state_table:
        return []
    try:
        resp = _system_state_table.get_item(Key={"pk": "tradeEvents"})
        item = resp.get("Item") or {}
        events = item.get("events") or []
        return events[: max(1, min(int(limit), 200))]
    except Exception:
        return []


def add_to_linegroups(target_id: str, target_type: str = "group") -> None:
    if not _line_groups_table:
        return
    _line_groups_table.put_item(
        Item={
            "targetId": target_id,
            "type": target_type,
            "createdAt": _now_ms(),
        }
    )


def remove_from_linegroups(target_id: str) -> None:
    if not _line_groups_table:
        return
    _line_groups_table.delete_item(Key={"targetId": target_id})


def is_target_whitelisted(target_id: str) -> bool:
    # If table not configured, treat as allowed
    if not _line_groups_table:
        return True
    resp = _line_groups_table.get_item(Key={"targetId": target_id})
    return "Item" in resp


# Backward-compatible aliases
def add_group_to_whitelist(group_id: str, group_type: str = "group") -> None:
    add_to_linegroups(group_id, group_type)


def remove_group_from_whitelist(group_id: str) -> None:
    remove_from_linegroups(group_id)


def is_group_whitelisted(group_id: str) -> bool:
    return is_target_whitelisted(group_id)
