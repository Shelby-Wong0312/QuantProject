"""
Lightweight publisher for trade events to SNS.

Usage:
    from publisher import publish_trade_event
    publish_trade_event(
        topic_arn="arn:aws:sns:ap-northeast-1:123456789012:trade-events-topic",
        symbol="BTCUSDT", side="BUY", price=50000.5, qty=0.01,
        source="mybot", note="entry"
    )
"""

import json
from typing import Any, Optional

import boto3


def publish_trade_event(
    *,
    topic_arn: str,
    symbol: str,
    side: str,
    price: Optional[float] = None,
    qty: Optional[float] = None,
    source: Optional[str] = None,
    note: Optional[str] = None,
    **extra: Any,
):
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

    sns = boto3.client("sns")
    return sns.publish(TopicArn=topic_arn, Message=json.dumps(payload))
