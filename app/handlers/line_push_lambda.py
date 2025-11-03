import json
import os
import logging
import uuid
import time
import random
from typing import Any, Dict, List, Optional

import boto3

try:
    from linebot import LineBotApi
    from linebot.models import TextSendMessage
    from linebot.exceptions import LineBotApiError
except Exception:
    LineBotApi = None  # type: ignore
    TextSendMessage = None  # type: ignore
    LineBotApiError = Exception  # type: ignore


logger = logging.getLogger(__name__)
_TRACE_ID: str = ""


def _log(event_type: str, status: str, **fields):
    rec = {
        "trace_id": _TRACE_ID or str(uuid.uuid4()),
        "event_type": event_type,
        "status": status,
    }
    rec.update(fields)
    try:
        logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        logger.info(str(rec))


def _get_param(name: str) -> Optional[str]:
    if not name:
        return None
    try:
        ssm = boto3.client("ssm")
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
        return resp.get("Parameter", {}).get("Value")
    except Exception:
        _log("ssm_get_parameter", "error", name=name)
        return None


def _get_line_client() -> "LineBotApi":
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        token = _get_param(os.getenv("LINE_CHANNEL_ACCESS_TOKEN_PARAM", ""))
    if not token:
        raise RuntimeError("LINE access token not configured")
    if LineBotApi is None or TextSendMessage is None:
        raise RuntimeError("line-bot-sdk not available")
    # Set reasonable timeout (seconds)
    return LineBotApi(token, timeout=5)


def _ddb_table(name_env: str, fallback_env: Optional[str] = None):
    table_name = os.getenv(name_env) or (
        os.getenv(fallback_env) if fallback_env else None
    )
    if not table_name:
        raise RuntimeError(f"Missing DynamoDB table env: {name_env}")
    return boto3.resource("dynamodb").Table(table_name)


def _format_event_zh(evt: Dict[str, Any]) -> str:
    sym = evt.get("symbol") or evt.get("ticker") or "?"
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
    src = evt.get("source")
    note = evt.get("note") or evt.get("message")
    parts = []
    if src:
        parts.append(f"[{src}]")
    parts.append(str(sym))
    if side:
        parts.append(str(side))
    if qty is not None:
        parts.append(str(qty))
    if price is not None:
        parts.append(f"@ {price}")
    if note:
        parts.append(f"｜{note}")
    return "交易事件：" + " ".join([p for p in parts if p])


def _format_trade_report(evt: Dict[str, Any]) -> str:
    side_raw = (evt.get("side") or evt.get("action") or "").upper()
    qty = evt.get("qty") or evt.get("quantity")
    symbol = evt.get("symbol") or evt.get("ticker") or "?"
    price = evt.get("price")
    status = evt.get("status") or ""
    pnl = evt.get("pnl")
    deal_id = evt.get("dealId")

    header = [side_raw]
    if qty is not None:
        header.append(str(qty))
    header.append(str(symbol))
    if price is not None:
        header.append(f"@ {price}")
    lines = [
        "【交易回報提交】",
        " ".join(header),
        f"狀態: {status}",
    ]
    if pnl is not None:
        lines.append(f"PnL: {pnl}")
    if deal_id:
        lines.append(f"Deal: {deal_id}")
    return "\n".join(lines)


def _list_targets() -> List[str]:
    groups_tbl = _ddb_table("GROUPS_TABLE", "LINE_GROUPS_TABLE")
    ids: List[str] = []
    start_key = None
    while True:
        if start_key:
            resp = groups_tbl.scan(
                ExclusiveStartKey=start_key, ProjectionExpression="targetId"
            )
        else:
            resp = groups_tbl.scan(ProjectionExpression="targetId")
        ids.extend(
            [it.get("targetId") for it in resp.get("Items", []) if it.get("targetId")]
        )
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break

    return ids


def _is_whitelisted(target_id: str) -> bool:
    try:
        groups_tbl = _ddb_table("GROUPS_TABLE", "LINE_GROUPS_TABLE")
        resp = groups_tbl.get_item(Key={"targetId": target_id})
        return "Item" in resp
    except Exception:
        # If table missing, default allow
        return True


def _put_system_summary(status: Dict[str, Any]) -> None:
    state_tbl = _ddb_table("STATE_TABLE", "SYSTEM_STATE_TABLE")
    item = {"pk": "summary", "updatedAt": int(__import__("time").time() * 1000)}
    item.update(status)
    state_tbl.put_item(Item=item)


def _put_system_positions(snap: Dict[str, Any]) -> None:
    state_tbl = _ddb_table("STATE_TABLE", "SYSTEM_STATE_TABLE")
    item = {"pk": "positions", "updatedAt": int(__import__("time").time() * 1000)}
    item.update(snap)
    state_tbl.put_item(Item=item)


def _put_trade_event(evt: Dict[str, Any]) -> None:
    events_tbl = _ddb_table("EVENTS_TABLE")
    # Ensure required fields
    if "ts" not in evt:
        from datetime import datetime, timezone

        evt["ts"] = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    if "bucket" not in evt:
        evt["bucket"] = "ALL"
    events_tbl.put_item(Item=evt)


def lambda_handler(event, context):
    global _TRACE_ID
    _TRACE_ID = str(uuid.uuid4())
    _log(
        "lambda_start",
        "ok",
        records=len(event.get("Records", [])) if isinstance(event, dict) else 0,
    )
    client = _get_line_client()
    delivered = 0
    subs_total = 0
    targets = _list_targets()
    subs_total = len(targets)

    # Accumulate texts to push within this single invoke
    texts_to_push: List[str] = []

    records = event.get("Records", []) if isinstance(event, dict) else []
    for r in records:
        if r.get("EventSource") != "aws:sns" or "Sns" not in r:
            continue
        message = r["Sns"].get("Message", "")
        try:
            payload = json.loads(message)
        except Exception:
            _log("sns_message", "error", reason="invalid_json", sample=message[:200])
            continue
        if not isinstance(payload, dict):
            continue

        ptype = str(payload.get("type") or "event").lower()
        if ptype == "status" or any(
            k in payload
            for k in (
                "equity",
                "cash",
                "unrealizedPnL",
                "unrealized_pnl",
                "upnl",
                "realizedPnL",
                "realized_pnl",
                "rpnl",
            )
        ):
            acct = str(payload.get("accountId") or payload.get("account") or "default")
            status = {
                "equity": payload.get("equity"),
                "cash": payload.get("cash"),
                "unrealizedPnL": payload.get("unrealizedPnL")
                or payload.get("unrealized_pnl")
                or payload.get("upnl"),
                "realizedPnL": payload.get("realizedPnL")
                or payload.get("realized_pnl")
                or payload.get("rpnl"),
                "accountId": acct,
            }
            try:
                _put_system_summary(status)
                _log("system_summary", "ok")
            except Exception as e:
                _log("system_summary", "error", error=str(e))
            continue

        if ptype == "positions" or "positions" in payload:
            try:
                _put_system_positions(payload)
                _log("system_positions", "ok")
            except Exception as e:
                _log("system_positions", "error", error=str(e))
            continue

        # treat as trade event -> persist + enqueue for push
        try:
            _put_trade_event({**payload})
            _log("trade_event", "ok")
        except Exception as e:
            _log("trade_event", "error", error=str(e))
        # Choose template with detailed execution report if fields present
        if any(k in payload for k in ("status", "pnl", "dealId")):
            texts_to_push.append(_format_trade_report(payload))
        else:
            texts_to_push.append(_format_event_zh(payload))

    # Batch push per target with message chunks (LINE max 5 per push)
    if texts_to_push and targets:
        from linebot.models import TextSendMessage

        def _msg_chunks(lst: List[str], size: int = 5):
            for i in range(0, len(lst), size):
                yield lst[i : i + size]

        def _target_chunks(lst: List[str], size: int = 50):
            for i in range(0, len(lst), size):
                yield lst[i : i + size]

        for tchunk in _target_chunks(targets, 50):
            for rid in tchunk:
                ok_once = False
                for chunk in _msg_chunks(texts_to_push, 5):
                    msgs = [TextSendMessage(text=t) for t in chunk]
                    try:
                        client.push_message(rid, msgs)
                        ok_once = True
                        _log("push", "ok", target_id=rid, count=len(chunk))
                    except LineBotApiError as e:
                        status = getattr(e, "status_code", None)
                        if status in (429,) or (
                            isinstance(status, int) and 500 <= status < 600
                        ):
                            time.sleep(0.5 + random.random() * 0.5)
                            try:
                                client.push_message(rid, msgs)
                                ok_once = True
                                _log(
                                    "push_retry",
                                    "ok",
                                    target_id=rid,
                                    count=len(chunk),
                                    status=status,
                                )
                            except Exception as e2:
                                _log(
                                    "push_retry",
                                    "error",
                                    target_id=rid,
                                    error=str(e2),
                                    status=status,
                                )
                        else:
                            _log("push", "error", target_id=rid, error=str(e))
                    except Exception as e:
                        _log("push", "error", target_id=rid, error=str(e))
                if ok_once:
                    delivered += 1
            # small gap between batches to be gentle to LINE API
            time.sleep(0.2)

    _log("lambda_end", "ok", delivered=delivered, targets=subs_total)
    return {"ok": True, "delivered": delivered, "subscribers": subs_total}
