import logging
import os
from datetime import datetime, timezone
from typing import Optional

import boto3
from decimal import Decimal

logger = logging.getLogger(__name__)

AWS_REGION_ENV = "AWS_REGION"
STATE_TABLE_ENV = "STATE_TABLE"
EVENTS_TABLE_ENV = "EVENTS_TABLE"
DEFAULT_REGION = "ap-northeast-1"
DEFAULT_STATE_TABLE = "SystemState"
DEFAULT_EVENTS_TABLE = "TradeEvents"


def _ddb_resource():
    region = os.getenv(AWS_REGION_ENV, DEFAULT_REGION)
    return boto3.resource("dynamodb", region_name=region)


def write_summary(equity: float, cash: float, upnl: float, rpnl: float) -> None:
    table_name = os.getenv(STATE_TABLE_ENV, DEFAULT_STATE_TABLE)
    table = _ddb_resource().Table(table_name)
    item = {
        "pk": "summary",
        "updatedAt": int(datetime.now(timezone.utc).timestamp() * 1000),
        "equity": Decimal(str(equity)),
        "cash": Decimal(str(cash)),
        "unrealizedPnL": Decimal(str(upnl)),
        "realizedPnL": Decimal(str(rpnl)),
    }
    table.put_item(Item=item)
    logger.info("Wrote summary state to %s", table_name)


def write_positions_text(text: str) -> None:
    table_name = os.getenv(STATE_TABLE_ENV, DEFAULT_STATE_TABLE)
    table = _ddb_resource().Table(table_name)
    item = {
        "pk": "positions",
        "updatedAt": int(datetime.now(timezone.utc).timestamp() * 1000),
        "positionsText": text,
    }
    table.put_item(Item=item)
    logger.info("Wrote positions text to %s", table_name)


def append_trade_event(
    *,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    ts: Optional[str] = None,
) -> None:
    events_table_name = os.getenv(EVENTS_TABLE_ENV, DEFAULT_EVENTS_TABLE)
    events_table = _ddb_resource().Table(events_table_name)

    if ts is None:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    item = {
        "pk": "trade",
        "sk": ts,
        "symbol": symbol,
        "side": side,
        "quantity": Decimal(str(quantity)),
        "price": Decimal(str(price)),
    }
    events_table.put_item(Item=item)
    logger.info("Appended trade event to %s", events_table_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    write_summary(equity=100000.0, cash=55000.0, upnl=1234.56, rpnl=-321.0)
    write_positions_text("BTCUSDT +0.5 @ 50000.0\nXAUUSD -1 @ 2405.0")
    append_trade_event(symbol="BTCUSDT", side="BUY", quantity=0.5, price=50000.0)
    print("State snapshot and trade event written to DynamoDB")
