import base64
import hashlib
import hmac
import json
import os
import logging
import uuid
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import boto3

try:
    from linebot import LineBotApi
    from linebot.models import TextSendMessage
    from linebot import WebhookParser
    from linebot.exceptions import InvalidSignatureError
    from linebot.exceptions import LineBotApiError
except Exception:
    LineBotApi = None  # type: ignore
    TextSendMessage = None  # type: ignore
    WebhookParser = None  # type: ignore
    InvalidSignatureError = Exception  # type: ignore
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
        return None


def _get_line_client() -> "LineBotApi":
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        token = _get_param(os.getenv("LINE_CHANNEL_ACCESS_TOKEN_PARAM", ""))
    if not token:
        raise RuntimeError("LINE access token not configured")
    if LineBotApi is None or TextSendMessage is None:
        raise RuntimeError("line-bot-sdk not available")
    return LineBotApi(token, timeout=5)


def _get_secret() -> Optional[str]:
    sec = os.getenv("LINE_CHANNEL_SECRET")
    if not sec:
        sec = _get_param(os.getenv("LINE_CHANNEL_SECRET_PARAM", ""))
    return sec


def _verify_signature(secret: str, headers: Dict[str, str], raw: bytes) -> bool:
    # Prefer official WebhookParser when available
    try:
        sig_header = None
        for k, v in (headers or {}).items():
            if k.lower() == "x-line-signature":
                sig_header = v
                break
        if not sig_header:
            return False
        if WebhookParser is not None:
            parser = WebhookParser(secret)
            # We only use parser for signature validation; event objects are not required here
            try:
                parser.parse(raw.decode("utf-8"), sig_header)
                return True
            except InvalidSignatureError:
                return False
        # Fallback to manual HMAC check
        digest = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).digest()
        signature = base64.b64encode(digest).decode("utf-8")
        return hmac.compare_digest(signature, sig_header)
    except Exception:
        _log("signature", "error", reason="verification_failed")
        return False


def _ddb_table(name_env: str, fallback_env: Optional[str] = None):
    table_name = os.getenv(name_env) or (
        os.getenv(fallback_env) if fallback_env else None
    )
    if not table_name:
        raise RuntimeError(f"Missing DynamoDB table env: {name_env}")
    return boto3.resource("dynamodb").Table(table_name)


def _extract_recipient(source: Dict[str, Any]) -> Tuple[str, str]:
    if not source:
        return "", ""
    if "userId" in source:
        return source["userId"], "user"
    if "groupId" in source:
        return source["groupId"], "group"
    if "roomId" in source:
        return source["roomId"], "room"
    return "", source.get("type", "")


def _help_text() -> str:
    return (
        "可用指令:\n"
        "/help - 顯示說明\n"
        "/subscribe - 訂閱交易推播\n"
        "/unsubscribe - 取消訂閱\n"
        "/last [n] - 查詢最近 n 筆事件 (預設 5)\n"
        "/status [accountId] - 查詢帳戶狀態摘要\n"
        "/positions [accountId] - 回報持倉摘要\n"
        "/ping - 偵測連線\n"
    )


def _is_whitelisted(target_id: str) -> bool:
    try:
        groups_tbl = _ddb_table("GROUPS_TABLE", "LINE_GROUPS_TABLE")
        resp = groups_tbl.get_item(Key={"targetId": target_id})
        return "Item" in resp
    except Exception:
        # If whitelist table not configured, allow by default
        return True


def _format_event(evt: Dict[str, Any]) -> str:
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


def _get_last_events(n: int) -> List[Dict[str, Any]]:
    """Query latest N trade events from TradeEvents via GSI (desc by ts)."""
    try:
        events_tbl = _ddb_table("EVENTS_TABLE")
        from boto3.dynamodb.conditions import Key

        resp = events_tbl.query(
            IndexName="recent-index",
            KeyConditionExpression=Key("bucket").eq("ALL"),
            ScanIndexForward=False,
            Limit=max(1, min(n, 50)),
        )
        return resp.get("Items", [])
    except Exception:
        return []


def _get_status() -> Optional[Dict[str, Any]]:
    """Read SystemState summary snapshot (pk='summary')."""
    try:
        state_tbl = _ddb_table("STATE_TABLE", "SYSTEM_STATE_TABLE")
        res = state_tbl.get_item(Key={"pk": "summary"})
        return res.get("Item")
    except Exception:
        logger.exception("Failed to read SystemState summary")
        return None


def _get_positions() -> Optional[Dict[str, Any]]:
    """Read SystemState positions snapshot (pk='positions')."""
    try:
        state_tbl = _ddb_table("STATE_TABLE", "SYSTEM_STATE_TABLE")
        res = state_tbl.get_item(Key={"pk": "positions"})
        return res.get("Item")
    except Exception:
        logger.exception("Failed to read SystemState positions")
        return None


def _handle_text(
    client: "LineBotApi",
    text: str,
    reply_token: str,
    recipient_id: str,
    recipient_type: str,
) -> None:
    raw = text.strip()
    if raw.startswith("/"):
        raw = raw[1:]
    parts = raw.split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]

    if cmd in ("help", "h", "?"):
        _reply_with_retry(client, reply_token, _help_text())
        return

    if cmd in ("ping",):
        _reply_with_retry(client, reply_token, "pong")
        return

    if cmd in ("subscribe", "sub"):
        if not recipient_id:
            _reply_with_retry(
                client, reply_token, "無法取得聊天室 ID，請在正式對話中使用。"
            )
            return
        subs_tbl = _ddb_table("SUBSCRIBERS_TABLE")
        subs_tbl.put_item(
            Item={
                "recipientId": recipient_id,
                "type": recipient_type,
                "createdAt": int(__import__("time").time() * 1000),
            }
        )
        try:
            groups_tbl = _ddb_table("GROUPS_TABLE", "LINE_GROUPS_TABLE")
            groups_tbl.put_item(
                Item={
                    "targetId": recipient_id,
                    "type": recipient_type,
                    "createdAt": int(__import__("time").time() * 1000),
                }
            )
        except Exception:
            pass
        _reply_with_retry(client, reply_token, "✅ 已訂閱交易推播")
        return

    if cmd in ("unsubscribe", "unsub"):
        if recipient_id:
            subs_tbl = _ddb_table("SUBSCRIBERS_TABLE")
            subs_tbl.delete_item(Key={"recipientId": recipient_id})
        _reply_with_retry(client, reply_token, "🛑 已取消訂閱")
        return

    if cmd in ("last", "l"):
        try:
            if recipient_id and not _is_whitelisted(recipient_id):
                _reply_with_retry(
                    client, reply_token, "此對話未在白名單，請輸入 /subscribe 以授權"
                )
                return
        except Exception:
            pass
        try:
            n = int(args[0]) if args else 5
        except Exception:
            n = 5
        events = _get_last_events(n)
        if not events:
            _reply_with_retry(client, reply_token, "目前尚無交易事件")
            return
        lines = [_format_event(e) for e in events]
        _reply_with_retry(client, reply_token, "\n".join(lines))
        return

    if cmd in ("status",):
        try:
            if recipient_id and not _is_whitelisted(recipient_id):
                _reply_with_retry(
                    client, reply_token, "此對話未在白名單，請輸入 /subscribe 以授權"
                )
                return
        except Exception:
            pass
        st = _get_status()
        if not st:
            _reply_with_retry(client, reply_token, "尚無帳戶狀態（summary）")
            return

        def _fmt2(v):
            try:
                return f"{float(v):.2f}"
            except Exception:
                return str(v)

        parts = [
            f"帳戶：{st.get('accountId') or 'default'}",
            (
                f"Equity：{_fmt2(st.get('equity'))}"
                if st.get("equity") is not None
                else None
            ),
            f"Cash：{_fmt2(st.get('cash'))}" if st.get("cash") is not None else None,
            (
                f"未實現PnL：{_fmt2(st.get('unrealizedPnL'))}"
                if st.get("unrealizedPnL") is not None
                else None
            ),
            (
                f"已實現PnL：{_fmt2(st.get('realizedPnL'))}"
                if st.get("realizedPnL") is not None
                else None
            ),
        ]
        body = "狀態摘要\n" + "\n".join([p for p in parts if p])
        _reply_with_retry(client, reply_token, body)
        return

    if cmd in ("positions", "pos"):
        try:
            if recipient_id and not _is_whitelisted(recipient_id):
                _reply_with_retry(
                    client, reply_token, "此對話未在白名單，請輸入 /subscribe 以授權"
                )
                return
        except Exception:
            pass
        snap = _get_positions()
        if not snap:
            _reply_with_retry(client, reply_token, "尚無持倉資料（positions）")
            return
        text_val = snap.get("text") or snap.get("message") or snap.get("content")
        if not text_val:
            _reply_with_retry(client, reply_token, "尚無持倉文字內容")
            return
        _reply_with_retry(client, reply_token, str(text_val))
        return

    _reply_with_retry(client, reply_token, "未知指令，輸入 /help")


def _reply_with_retry(client: "LineBotApi", reply_token: str, text: str) -> None:
    try:
        client.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError as e:
        status = getattr(e, "status_code", None)
        if status in (429,) or (isinstance(status, int) and 500 <= status < 600):
            time.sleep(0.5 + random.random() * 0.5)
            try:
                client.reply_message(reply_token, TextSendMessage(text=text))
            except Exception as e2:
                _log("reply_retry", "error", error=str(e2), status=status)
        else:
            _log("reply", "error", error=str(e))
    except Exception as e:
        _log("reply", "error", error=str(e))


def lambda_handler(event, context):
    # Verify signature
    headers = event.get("headers") or {}
    body_str = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(body_str)
    else:
        raw = body_str.encode("utf-8")

    secret = _get_secret()
    global _TRACE_ID
    _TRACE_ID = str(uuid.uuid4())
    _log("lambda_start", "ok")
    if not secret or not _verify_signature(secret, headers, raw):
        return {
            "statusCode": 403,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": False, "error": "Invalid signature"}),
        }

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        _log("webhook", "error", reason="invalid_json", error=str(e))
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": False, "error": "Invalid JSON"}),
        }

    client = _get_line_client()
    for ev in payload.get("events", []):
        etype = ev.get("type")
        if etype == "join":
            src = ev.get("source") or {}
            gid = src.get("groupId") or src.get("roomId")
            gtype = (
                "group"
                if src.get("groupId")
                else ("room" if src.get("roomId") else src.get("type", ""))
            )
            if gid:
                try:
                    groups_tbl = _ddb_table("GROUPS_TABLE", "LINE_GROUPS_TABLE")
                    groups_tbl.put_item(
                        Item={
                            "targetId": gid,
                            "type": gtype,
                            "createdAt": int(__import__("time").time() * 1000),
                        }
                    )
                    _log("join", "ok", target_id=gid, group_type=gtype)
                except Exception as e:
                    _log("join", "error", target_id=gid, error=str(e))
            rt = ev.get("replyToken")
            if rt:
                client.reply_message(
                    rt,
                    TextSendMessage(text="已加入群組，若要接收推播請輸入 /subscribe"),
                )
            continue

        if etype == "leave":
            src = ev.get("source") or {}
            gid = src.get("groupId") or src.get("roomId")
            if gid:
                try:
                    groups_tbl = _ddb_table("GROUPS_TABLE", "LINE_GROUPS_TABLE")
                    groups_tbl.delete_item(Key={"targetId": gid})
                    _log("leave", "ok", target_id=gid)
                except Exception as e:
                    _log("leave", "error", target_id=gid, error=str(e))
            continue

        if etype == "message" and (ev.get("message", {}).get("type") == "text"):
            reply_token = ev.get("replyToken")
            recipient_id, recipient_type = _extract_recipient(ev.get("source") or {})
            text = ev.get("message", {}).get("text", "")
            if reply_token:
                _handle_text(client, text, reply_token, recipient_id, recipient_type)

    _log("lambda_end", "ok")
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"ok": True}),
    }
