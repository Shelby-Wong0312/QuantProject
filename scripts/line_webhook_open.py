import os, json, redis, requests
from datetime import datetime, timezone, timedelta
from flask import Flask, request

TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
r = redis.StrictRedis.from_url(os.getenv("REDIS_URL","redis://redis:6379/0"), decode_responses=True)
SUB_KEY = "line:subs"
STATUS_JSON = "/app/state/capital_api_status.json"  #  改成共用目錄
MAX_AGE_MIN = 15

app = Flask(__name__)

def is_fresh(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)) <= timedelta(minutes=MAX_AGE_MIN)
    except Exception:
        return False

def summary_text():
    try:
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        ts = d.get("timestamp","(none)")
        res = d.get("results",{})
        ok = lambda k: ("OK" if res.get(k) else "FAIL")
        return (
            "[System Status]\n"
            f"Auth:{ok('authentication')} Docs:{ok('documentation')} WS:{ok('websocket')}\n"
            f"Demo:{ok('demo_api')} Live:{ok('live_api')}\n"
            f"Updated:{ts} ({'fresh' if is_fresh(ts) else 'stale'})"
        )
    except Exception:
        return "[System Status] file missing"

def reply(token, text):
    if not TOKEN or not token: return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    data = {"replyToken": token, "messages":[{"type":"text","text": text[:4900]}]}
    try: requests.post(url, headers=headers, json=data, timeout=10)
    except Exception: pass

@app.route("/line/webhook", methods=["GET","HEAD","POST"])
@app.route("/callback",      methods=["GET","HEAD","POST"])
def webhook():
    if request.method in ("GET","HEAD"): return "OK", 200
    body = request.get_json(silent=True) or {}
    for e in body.get("events",[]):
        if e.get("type")=="message" and e.get("message",{}).get("type")=="text":
            text = (e["message"]["text"] or "").strip().lower()
            token = e.get("replyToken")
            src = e.get("source",{})
            chat_id = src.get("groupId") or src.get("roomId") or src.get("userId")
            if text in ("/help","help"):
                reply(token, "Commands:\n/summary\n/status\n/subscribe\n/unsubscribe\n/id\n/help")
            elif text in ("/summary","/status","summary","status"):
                reply(token, summary_text())
            elif text.startswith("/subscribe"):
                r.sadd(SUB_KEY, chat_id); reply(token, "Subscribed.")
            elif text.startswith("/unsubscribe"):
                r.srem(SUB_KEY, chat_id); reply(token, "Unsubscribed.")
            elif text in ("/id","id"):
                reply(token, f"ID: {chat_id}")
    return "OK", 200

@app.route("/healthz")
def healthz(): return "ok", 200
