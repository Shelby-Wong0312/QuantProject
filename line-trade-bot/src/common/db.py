import os
import time
from decimal import Decimal
from typing import List, Optional, Dict, Any

import boto3
from boto3.dynamodb.conditions import Key


_dynamodb = boto3.resource("dynamodb")
_subs_table = _dynamodb.Table(os.environ.get("SUBSCRIBERS_TABLE", ""))
_events_table = _dynamodb.Table(os.environ.get("EVENTS_TABLE", ""))


def _now_ms() -> int:
    return int(time.time() * 1000)


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
    # Ensure pk and ts exist
    if "pk" not in item:
        item["pk"] = "EVENT"
    if "ts" not in item:
        item["ts"] = _now_ms()
    # Convert float numbers to Decimal
    for k, v in list(item.items()):
        if isinstance(v, float):
            item[k] = Decimal(str(v))
    _events_table.put_item(Item=item)


def get_last_events(limit: int = 5) -> List[Dict[str, Any]]:
    resp = _events_table.query(
        KeyConditionExpression=Key("pk").eq("EVENT"),
        ScanIndexForward=False,
        Limit=max(1, min(limit, 50)),
    )
    return resp.get("Items", [])
