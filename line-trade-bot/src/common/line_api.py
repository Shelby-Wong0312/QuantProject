import json
import os
import sys
import urllib.request
import urllib.error
import time
import random


LINE_API_BASE = "https://api.line.me/v2/bot"


def _auth_header():
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Missing LINE_CHANNEL_ACCESS_TOKEN in environment")
    return {"Authorization": f"Bearer {token}"}


def _post_json(url: str, payload: dict, timeout: int = 5) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(_auth_header())
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    def _attempt():
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            if not data:
                return {"status": resp.status}
            return json.loads(data.decode("utf-8"))

    try:
        return _attempt()
    except urllib.error.HTTPError as e:
        status = getattr(e, "code", 0)
        retriable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
        if retriable:
            time.sleep(0.5 + random.random() * 0.5)
            try:
                return _attempt()
            except Exception as e2:
                try:
                    err_body = e2.read().decode("utf-8")  # type: ignore
                except Exception:
                    err_body = str(e2)
                return {"error": True, "status": getattr(e2, "code", status), "body": err_body}
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        return {"error": True, "status": status, "body": err_body}
    except Exception as e:
        return {"error": True, "status": 0, "body": str(e)}


def reply_message(reply_token: str, text: str) -> dict:
    url = f"{LINE_API_BASE}/message/reply"
    payload = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": text[:4900],
            }
        ],
    }
    return _post_json(url, payload)


def push_message(to: str, text: str) -> dict:
    url = f"{LINE_API_BASE}/message/push"
    payload = {
        "to": to,
        "messages": [
            {
                "type": "text",
                "text": text[:4900],
            }
        ],
    }
    return _post_json(url, payload)


def get_profile(user_id: str) -> dict:
    # Works for 1:1 userId. For groups/rooms, use member profile endpoints (not used here).
    headers = _auth_header()
    req = urllib.request.Request(
        f"{LINE_API_BASE}/profile/{user_id}", headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
            return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"error": True, "message": str(e)}
