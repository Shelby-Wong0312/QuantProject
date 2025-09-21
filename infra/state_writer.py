# infra/state_writer.py
import os, time
from decimal import Decimal
import boto3
import logging

STATE_TABLE = os.environ.get("STATE_TABLE", "SystemState")
EVENTS_TABLE = os.environ.get("EVENTS_TABLE", "TradeEvents")
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-northeast-1"

logging.basicConfig(level=os.getenv("STATE_WRITER_LOG","INFO").upper())
log = logging.getLogger("infra.state_writer")

ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
state_table = ddb.Table(STATE_TABLE)
events_table = ddb.Table(EVENTS_TABLE)

def write_summary(equity: float, cash: float, upnl: float, rpnl: float):
    state_table.put_item(Item={
        "pk":"summary",
        "equity": float(equity),
        "cash":   float(cash),
        "upnl":   float(upnl),
        "rpnl":   float(rpnl),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    })
    log.info("Wrote summary state to %s", STATE_TABLE)

def write_positions_text(text: str):
    state_table.put_item(Item={
        "pk":"positions",
        "text": str(text),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    })
    log.info("Wrote positions text to %s", STATE_TABLE)

def append_trade_event(symbol: str, side: str, quantity: float, price: float, pnl: float | None = None):
    item = {
        "pk":"global",                                  # 保持與原 /last N 相容
        "ts": str(int(time.time()*1000)),               # 你的表要求字串型別
        "symbol": str(symbol),
        "side": str(side),
        "quantity": Decimal(str(quantity)),
        "price": Decimal(str(price)),
    }
    if pnl is not None:
        item["pnl"] = Decimal(str(pnl))
    events_table.put_item(Item=item)
    log.info("Appended trade event to %s", EVENTS_TABLE)
