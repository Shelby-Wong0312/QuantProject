"""
Infra Bridge: publish trade event to SNS.

Usage:
    from app.infra_bridge.event_bus import publish_trade_event

    # Requires environment variable TRADE_EVENTS_TOPIC_ARN
    publish_trade_event(
        symbol="BTCUSDT", side="BUY", price=50000.5, qty=0.01,
        source="mybot", note="entry"
    )

TopicArn is read from environment variable `TRADE_EVENTS_TOPIC_ARN`.
Region defaults to environment or SDK default; override via `region_name`.
"""

import json
import os
from typing import Any, Optional

import boto3


def publish_trade_event(
    *,
    symbol: str,
    side: str,
    price: Optional[float] = None,
    qty: Optional[float] = None,
    source: Optional[str] = None,
    note: Optional[str] = None,
    region_name: Optional[str] = None,
    **extra: Any,
):
    arn = os.getenv("TRADE_EVENTS_TOPIC_ARN")
    if not arn:
        raise ValueError("Missing TRADE_EVENTS_TOPIC_ARN in environment")

    payload = {"symbol": symbol, "side": side}
    if price is not None:
        payload["price"] = float(price)
    if qty is not None:
        payload["qty"] = qty
    if source:
        payload["source"] = source
    if note:
        payload["note"] = note
    payload.update(extra)

    sns = boto3.client("sns", region_name=region_name)
    return sns.publish(TopicArn=arn, Message=json.dumps(payload))
