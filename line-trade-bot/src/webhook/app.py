import base64
import hashlib
import hmac
import json
import os
from typing import Dict, Any

from common import db
from common.line_api import reply_message
from common.ingest_core import format_status_zh, format_positions_zh


def _resp(status: int, body: Any = None) -> Dict[str, Any]:
    if body is None:
        body = {"ok": status < 400}
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": body}


def _verify_line_signature(headers: Dict[str, str], raw_body: bytes) -> bool:
    secret = None
    try:
        from common.secrets import get_param, get_secret
        secret = get_param(os.environ.get("LINE_CHANNEL_SECRET_PARAM", "")) or \
                 get_secret(os.environ.get("LINE_CHANNEL_SECRET_SECRET_ID", ""))
    except Exception:
        secret = None
    if not secret:
        secret = os.environ.get("LINE_CHANNEL_SECRET", "")
    if not secret:
        # Secret is required for verification
        return False
    sig_header = None
    for k, v in (headers or {}).items():
        if k.lower() == "x-line-signature":
            sig_header = v
            break
    if not sig_header:
        return False
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(signature, sig_header)


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


def _extract_recipient(source: Dict[str, Any]) -> (str, str):
    if not source:
        return "", ""
    if "userId" in source:
        return source["userId"], "user"
    if "groupId" in source:
        return source["groupId"], "group"
    if "roomId" in source:
        return source["roomId"], "room"
    return "", source.get("type", "")


def _handle_text(text: str, reply_token: str, recipient_id: str, recipient_type: str) -> None:
    raw = text.strip()
    if raw.startswith("/"):
        raw = raw[1:]
    parts = raw.split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]

    if cmd in ("help", "h", "?"):
        reply_message(reply_token, _help_text())
        return

    if cmd in ("ping",):
        reply_message(reply_token, "pong")
        return

    if cmd in ("subscribe", "sub"):
        if not recipient_id:
            reply_message(reply_token, "無法取得聊天室 ID，請在正式對話中使用。")
            return
        db.register_subscriber(recipient_id, recipient_type or "unknown")
        try:
            db.add_to_linegroups(recipient_id, recipient_type or "unknown")
        except Exception:
            pass
        reply_message(reply_token, "✅ 已訂閱交易推播")
        return

    if cmd in ("unsubscribe", "unsub"):
        if recipient_id:
            db.unregister_subscriber(recipient_id)
        reply_message(reply_token, "🛑 已取消訂閱")
        return

    # For query-type commands, ensure whitelist (if configured)
    if cmd in ("last", "l"):
        try:
            if recipient_id and not db.is_target_whitelisted(recipient_id):
                reply_message(reply_token, "此對話未在白名單，請輸入 /subscribe 以授權")
                return
        except Exception:
            pass
        try:
            n = int(args[0]) if args else 5
        except Exception:
            n = 5
        # Prefer SystemState cache for performance
        events = db.get_trade_events_recent(limit=n)
        if not events:
            events = db.get_last_events(limit=n)
        if not events:
            reply_message(reply_token, "目前尚無交易事件")
            return
        lines = [_format_event(e) for e in events]
        reply_message(reply_token, "\n".join(lines))
        return

    if cmd in ("status",):
        try:
            if recipient_id and not db.is_target_whitelisted(recipient_id):
                reply_message(reply_token, "此對話未在白名單，請輸入 /subscribe 以授權")
                return
        except Exception:
            pass
        account_id = args[0] if args else "default"
        st = None
        if not args:
            try:
                st = db.get_system_summary()
            except Exception:
                st = None
        if not st:
            st = db.get_latest_status(account_id)
        if not st:
            reply_message(reply_token, f"尚無帳戶狀態（{account_id}）")
            return
        reply_message(reply_token, format_status_zh(st))
        return

    if cmd in ("positions", "pos"):
        try:
            if recipient_id and not db.is_target_whitelisted(recipient_id):
                reply_message(reply_token, "此對話未在白名單，請輸入 /subscribe 以授權")
                return
        except Exception:
            pass
        account_id = args[0] if args else "default"
        snap = None
        if not args:
            try:
                snap = db.get_system_positions()
            except Exception:
                snap = None
        if not snap:
            snap = db.get_latest_positions(account_id)
        if not snap:
            reply_message(reply_token, f"尚無持倉資料（{account_id}）")
            return
        reply_message(reply_token, format_positions_zh(snap))
        return

    reply_message(reply_token, "指令無效，輸入 /help 查看說明")
    return


def lambda_handler(event, context):
    headers = event.get("headers") or {}
    body_str = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(body_str)
    else:
        raw = body_str.encode("utf-8")

    if not _verify_line_signature(headers, raw):
        return _resp(403, {"ok": False, "error": "Invalid signature"})

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return _resp(400, {"ok": False, "error": "Invalid JSON"})

    for ev in payload.get("events", []):
        etype = ev.get("type")
        if etype == "join":
            src = ev.get("source") or {}
            gid = src.get("groupId") or src.get("roomId")
            gtype = "group" if src.get("groupId") else ("room" if src.get("roomId") else src.get("type", ""))
            if gid:
                try:
                    db.add_group_to_whitelist(gid, gtype)
                except Exception:
                    pass
            # Best effort welcome message
            rt = ev.get("replyToken")
            if rt:
                reply_message(rt, "已加入群組，若要接收推播請輸入 /subscribe")
            continue

        if etype == "leave":
            src = ev.get("source") or {}
            gid = src.get("groupId") or src.get("roomId")
            if gid:
                try:
                    db.remove_group_from_whitelist(gid)
                except Exception:
                    pass
            continue

        if etype == "message" and (ev.get("message", {}).get("type") == "text"):
            reply_token = ev.get("replyToken")
            recipient_id, recipient_type = _extract_recipient(ev.get("source") or {})
            text = ev.get("message", {}).get("text", "")
            _handle_text(text, reply_token, recipient_id, recipient_type)

    return _resp(200, {"ok": True})
