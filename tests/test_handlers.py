import base64
import hashlib
import hmac
import importlib
import json
import os
import sys
from types import ModuleType

import boto3
import pytest
from moto import mock_aws


def install_dummy_linebot(collected):
    """Install a dummy linebot SDK into sys.modules for tests.
    collected: dict to collect call logs.
    """
    lb = ModuleType("linebot")

    class DummyLineBotApi:
        def __init__(self, token: str):
            self.token = token

        def push_message(self, to, message):
            collected.setdefault("push", []).append(
                {"to": to, "text": getattr(message, "text", None)}
            )

        def reply_message(self, reply_token, message):
            collected.setdefault("reply", []).append(
                {"replyToken": reply_token, "text": getattr(message, "text", None)}
            )

    models = ModuleType("linebot.models")

    class TextSendMessage:
        def __init__(self, text: str):
            self.text = text

    lb.LineBotApi = DummyLineBotApi
    models.TextSendMessage = TextSendMessage

    sys.modules["linebot"] = lb
    sys.modules["linebot.models"] = models


@mock_aws
def test_line_push_trade_event_happy_path(monkeypatch):
    # Setup AWS resources
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
    ddb = boto3.client("dynamodb")
    ddb.create_table(
        TableName="trade-events-subscribers",
        KeySchema=[{"AttributeName": "recipientId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "recipientId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="TradeEvents",
        KeySchema=[{"AttributeName": "ts", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "ts", "AttributeType": "S"},
            {"AttributeName": "bucket", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "recent-index",
                "KeySchema": [
                    {"AttributeName": "bucket", "KeyType": "HASH"},
                    {"AttributeName": "ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="LineGroups",
        KeySchema=[{"AttributeName": "targetId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "targetId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="SystemState",
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    # seed subscriber and whitelist
    subs = boto3.resource("dynamodb").Table("trade-events-subscribers")
    subs.put_item(Item={"recipientId": "U1", "type": "user", "createdAt": 0})
    groups = boto3.resource("dynamodb").Table("LineGroups")
    groups.put_item(Item={"targetId": "U1", "type": "user", "createdAt": 0})

    # env
    os.environ.update(
        {
            "SUBSCRIBERS_TABLE": "trade-events-subscribers",
            "EVENTS_TABLE": "TradeEvents",
            "GROUPS_TABLE": "LineGroups",
            "STATE_TABLE": "SystemState",
            "LINE_CHANNEL_ACCESS_TOKEN": "dummy-token",
        }
    )

    # install dummy linebot
    calls = {}
    install_dummy_linebot(calls)

    # import handler
    import app.handlers.line_push_lambda as handler

    importlib.reload(handler)

    event = {
        "Records": [
            {
                "EventSource": "aws:sns",
                "Sns": {
                    "Message": json.dumps(
                        {
                            "symbol": "BTCUSDT",
                            "side": "BUY",
                            "price": 50000.5,
                            "qty": 0.01,
                            "source": "bot",
                        }
                    )
                },
            }
        ]
    }
    res = handler.lambda_handler(event, None)

    # assertions
    assert res.get("ok") is True
    assert res.get("delivered") == 1
    assert res.get("subscribers") == 1

    # Verify an item was written to TradeEvents
    events_tbl = boto3.resource("dynamodb").Table("TradeEvents")
    scan = events_tbl.scan()
    assert scan.get("Count", 0) >= 1


@mock_aws
def test_line_webhook_help_and_subscribe(monkeypatch):
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
    ddb = boto3.client("dynamodb")
    # tables
    ddb.create_table(
        TableName="trade-events-subscribers",
        KeySchema=[{"AttributeName": "recipientId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "recipientId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="LineGroups",
        KeySchema=[{"AttributeName": "targetId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "targetId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="SystemState",
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="TradeEvents",
        KeySchema=[{"AttributeName": "ts", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "ts", "AttributeType": "S"},
            {"AttributeName": "bucket", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "recent-index",
                "KeySchema": [
                    {"AttributeName": "bucket", "KeyType": "HASH"},
                    {"AttributeName": "ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    # env
    os.environ.update(
        {
            "SUBSCRIBERS_TABLE": "trade-events-subscribers",
            "GROUPS_TABLE": "LineGroups",
            "STATE_TABLE": "SystemState",
            "EVENTS_TABLE": "TradeEvents",
            "LINE_CHANNEL_ACCESS_TOKEN": "dummy-token",
            "LINE_CHANNEL_SECRET": "s3cr3t",
        }
    )

    calls = {}
    install_dummy_linebot(calls)
    import app.handlers.line_webhook_lambda as webhook

    importlib.reload(webhook)

    def make_event(text: str) -> Dict[str, Any]:
        body = json.dumps(
            {
                "events": [
                    {
                        "type": "message",
                        "replyToken": "r1",
                        "source": {"type": "user", "userId": "U9"},
                        "message": {"type": "text", "text": text},
                    }
                ]
            }
        )
        sig = base64.b64encode(
            hmac.new(
                os.environ["LINE_CHANNEL_SECRET"].encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return {
            "version": "2.0",
            "headers": {"X-Line-Signature": sig},
            "isBase64Encoded": False,
            "body": body,
        }

    # /help
    res = webhook.lambda_handler(make_event("/help"), None)
    assert res.get("statusCode") == 200
    assert len(calls.get("reply", [])) >= 1

    # /subscribe
    calls["reply"] = []
    res2 = webhook.lambda_handler(make_event("/subscribe"), None)
    assert res2.get("statusCode") == 200

    # verify records written
    subs_tbl = boto3.resource("dynamodb").Table("trade-events-subscribers")
    got = subs_tbl.get_item(Key={"recipientId": "U9"}).get("Item")
    assert got is not None
    groups_tbl = boto3.resource("dynamodb").Table("LineGroups")
    got2 = groups_tbl.get_item(Key={"targetId": "U9"}).get("Item")
    assert got2 is not None


@mock_aws
def test_line_webhook_status_and_last_three(monkeypatch):
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
    ddb = boto3.client("dynamodb")

    # Create tables
    ddb.create_table(
        TableName="trade-events-subscribers",
        KeySchema=[{"AttributeName": "recipientId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "recipientId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="LineGroups",
        KeySchema=[{"AttributeName": "targetId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "targetId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="SystemState",
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.create_table(
        TableName="TradeEvents",
        KeySchema=[{"AttributeName": "ts", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "ts", "AttributeType": "S"},
            {"AttributeName": "bucket", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "recent-index",
                "KeySchema": [
                    {"AttributeName": "bucket", "KeyType": "HASH"},
                    {"AttributeName": "ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    # Seed SystemState summary
    state_tbl = boto3.resource("dynamodb").Table("SystemState")
    state_tbl.put_item(
        Item={
            "pk": "summary",
            "accountId": "default",
            "equity": 100000,
            "cash": 60000,
            "unrealizedPnL": 800.5,
            "realizedPnL": -1200,
        }
    )

    # Seed LineGroups whitelist for user U9
    boto3.resource("dynamodb").Table("LineGroups").put_item(Item={"targetId": "U9"})

    # Seed TradeEvents 5 items
    ev_tbl = boto3.resource("dynamodb").Table("TradeEvents")
    base = "2025-01-01T12:00:00."
    for ms in ["001Z", "010Z", "020Z", "030Z", "040Z"]:
        ev_tbl.put_item(
            Item={
                "ts": base + ms,
                "bucket": "ALL",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "qty": 0.01,
                "price": 50000.0,
            }
        )

    # Env
    os.environ.update(
        {
            "SUBSCRIBERS_TABLE": "trade-events-subscribers",
            "GROUPS_TABLE": "LineGroups",
            "STATE_TABLE": "SystemState",
            "EVENTS_TABLE": "TradeEvents",
            "LINE_CHANNEL_ACCESS_TOKEN": "dummy-token",
            "LINE_CHANNEL_SECRET": "s3cr3t",
        }
    )

    calls = {}
    install_dummy_linebot(calls)
    import app.handlers.line_webhook_lambda as webhook

    importlib.reload(webhook)

    def signed_event(text: str) -> Dict[str, Any]:
        body = json.dumps(
            {
                "events": [
                    {
                        "type": "message",
                        "replyToken": "r1",
                        "source": {"type": "user", "userId": "U9"},
                        "message": {"type": "text", "text": text},
                    }
                ]
            }
        )
        sig = base64.b64encode(
            hmac.new(
                os.environ["LINE_CHANNEL_SECRET"].encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return {
            "version": "2.0",
            "headers": {"X-Line-Signature": sig},
            "isBase64Encoded": False,
            "body": body,
        }

    # /status should return 2-decimal numbers
    calls["reply"] = []
    res = webhook.lambda_handler(signed_event("/status"), None)
    assert res.get("statusCode") == 200
    reply_text = calls.get("reply", [{}])[-1].get("text", "")
    assert (
        "Equity" in reply_text and ": 100000.00" in reply_text or "Equity：100000.00" in reply_text
    )

    # /last 3 returns 3 lines
    calls["reply"] = []
    res2 = webhook.lambda_handler(signed_event("/last 3"), None)
    assert res2.get("statusCode") == 200
    text2 = calls.get("reply", [{}])[-1].get("text", "")
    assert len(text2.splitlines()) == 3
