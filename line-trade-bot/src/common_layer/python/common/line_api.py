import os
import time
import random
from typing import Optional

try:
    from linebot import LineBotApi
    from linebot.models import TextSendMessage
    from linebot.exceptions import LineBotApiError
except Exception:  # pragma: no cover - allow import without SDK for static checks
    LineBotApi = None  # type: ignore
    TextSendMessage = None  # type: ignore
    LineBotApiError = Exception  # type: ignore


_CLIENT: Optional["LineBotApi"] = None


def _get_client() -> "LineBotApi":
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    # Prefer SSM/Secrets Manager, fallback to env var
    token = None
    try:
        from .secrets import get_param, get_secret  # lazy import
        token = get_param(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN_PARAM", "")) or \
                get_secret(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN_SECRET_ID", ""))
    except Exception:
        token = None
    if not token:
        token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Missing LINE_CHANNEL_ACCESS_TOKEN in environment")
    if LineBotApi is None:
        raise RuntimeError("line-bot-sdk not available in runtime layer")
    _CLIENT = LineBotApi(token, timeout=5)
    return _CLIENT


def reply_message(reply_token: str, text: str) -> dict:
    client = _get_client()
    try:
        client.reply_message(reply_token, TextSendMessage(text=text[:4900]))
        return {"ok": True}
    except LineBotApiError as e:
        status = getattr(e, "status_code", None)
        if status in (429,) or (isinstance(status, int) and 500 <= status < 600):
            time.sleep(0.5 + random.random() * 0.5)
            try:
                client.reply_message(reply_token, TextSendMessage(text=text[:4900]))
                return {"ok": True, "retry": True}
            except Exception as e2:
                return {"error": True, "message": str(e2), "status": status}
        return {"error": True, "message": str(e), "status": status}
    except Exception as e:
        return {"error": True, "message": str(e)}


def push_message(to: str, text: str) -> dict:
    client = _get_client()
    try:
        client.push_message(to, TextSendMessage(text=text[:4900]))
        return {"ok": True}
    except LineBotApiError as e:
        status = getattr(e, "status_code", None)
        if status in (429,) or (isinstance(status, int) and 500 <= status < 600):
            time.sleep(0.5 + random.random() * 0.5)
            try:
                client.push_message(to, TextSendMessage(text=text[:4900]))
                return {"ok": True, "retry": True}
            except Exception as e2:
                return {"error": True, "message": str(e2), "status": status}
        return {"error": True, "message": str(e), "status": status}
    except Exception as e:
        return {"error": True, "message": str(e)}


def get_profile(user_id: str) -> dict:
    try:
        client = _get_client()
        prof = client.get_profile(user_id)
        # line-bot-sdk returns a Profile object; convert minimal fields
        return {
            "displayName": getattr(prof, "display_name", None) or getattr(prof, "displayName", None),
            "userId": getattr(prof, "user_id", None) or getattr(prof, "userId", None),
            "pictureUrl": getattr(prof, "picture_url", None) or getattr(prof, "pictureUrl", None),
            "statusMessage": getattr(prof, "status_message", None) or getattr(prof, "statusMessage", None),
        }
    except Exception as e:
        return {"error": True, "message": str(e)}
