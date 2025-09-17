# path: infra/publish_to_sns.py
import os, json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import boto3

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1")
TOPIC_ARN  = os.getenv("TRADE_EVENTS_TOPIC_ARN")  # 例: arn:aws:sns:ap-northeast-1:123456789012:trade-events
sns = boto3.client("sns", region_name=AWS_REGION)

def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def publish_trade_event(
    *,
    symbol: str,
    side: str,             # "buy" | "sell"
    quantity: float,
    price: float,
    status: str,           # "submitted" | "filled" | "closed" | "rejected"
    deal_id: Optional[str] = None,
    strategy: Optional[str] = None,
    pnl: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """把交易事件以 JSON 發到 SNS（line-push 會轉送到 LINE）。"""
    assert TOPIC_ARN, "Missing env TRADE_EVENTS_TOPIC_ARN"
    payload = {
        "timestamp": _iso(),
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "status": status,
        "dealId": deal_id,
        "strategy": strategy,
        "pnl": pnl,
        "extra": extra or {},
    }
    sns.publish(TopicArn=TOPIC_ARN, Message=json.dumps(payload))

if __name__ == "__main__":
    # 本地/EC2 快測：先 export TRADE_EVENTS_TOPIC_ARN 再跑
    publish_trade_event(symbol="US100", side="sell", quantity=2, price=18350.5,
                        status="filled", pnl=125.4, deal_id="ABC123")
    print("published.")
