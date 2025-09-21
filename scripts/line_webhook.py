import os, json, redis
from datetime import datetime, timezone, timedelta
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)
token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
secret = os.getenv("LINE_CHANNEL_SECRET")
redis_url = os.getenv("REDIS_URL","redis://redis:6379/0")
if not token or not secret:
    raise SystemExit("缺少 LINE_CHANNEL_ACCESS_TOKEN 或 LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(token)
handler = WebhookHandler(secret)
r = redis.StrictRedis.from_url(redis_url, decode_responses=True)
SUB_KEY = "line:subs"

STATUS_JSON = "/app/capital_api_status.json"
MAX_AGE_MIN = 15
VERIFY_MODE = os.getenv("LINE_VERIFY_MODE","0") == "1"   # ← 驗證模式

def _fresh(s):
    try: dt = datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception: return True
    return (datetime.now(timezone.utc)-dt.astimezone(timezone.utc)) <= timedelta(minutes=MAX_AGE_MIN)

def _summary_message():
    if not os.path.exists(STATUS_JSON):
        return "【系統狀態】找不到 capital_api_status.json"
    with open(STATUS_JSON,"r",encoding="utf-8") as f:
        d = json.load(f)
    res = d.get("results",{})
    ok = lambda k: "✅" if res.get(k) else "❌"
    ts = d.get("timestamp","(無)")
    return "\n".join([
        "【系統狀態 / System Status】",
        f"Auth: {ok('authentication')}  Docs: {ok('documentation')}  WS: {ok('websocket')}",
        f"Demo: {ok('demo_api')}  Live: {ok('live_api')}",
        f"Updated: {ts}  ({'新鮮' if _fresh(ts) else '可能過期'})",
        "",
        "指令：/summary /subscribe /unsubscribe /id /help（/status 同義）",
    ])

@app.route("/line/webhook", methods=["GET","HEAD","POST"])
@app.route("/callback",      methods=["GET","HEAD","POST"])
def cb():
    if request.method in ("GET","HEAD"):
        return "OK", 200
    if VERIFY_MODE:
        app.logger.warning("VERIFY_MODE=1: return 200 for POST (skip signature).")
        return "OK", 200
    body = request.get_data(as_text=True)
    sig  = request.headers.get("X-Line-Signature","")
    try:
        handler.handle(body, sig)
    except InvalidSignatureError:
        app.logger.warning("InvalidSignatureError: signature mismatch.")
        abort(400)
    return "OK", 200

@app.route("/healthz")
def healthz(): return "ok", 200

@handler.add(MessageEvent, message=TextMessage)
def on_msg(event):
    txt = (event.message.text or "").strip().lower()
    st = getattr(event.source,"type","user")
    if st=="group": chat_id = event.source.group_id
    elif st=="room": chat_id = event.source.room_id
    else: chat_id = event.source.user_id
    def reply(s): line_bot_api.reply_message(event.reply_token, TextSendMessage(text=s))
    if txt in ("/help","help"): reply("指令：/summary /subscribe /unsubscribe /id /help（/status 同義）"); return
    if txt in ("/summary","summary","/status","status"): reply(_summary_message()); return
    if txt.startswith("/subscribe"): r.sadd("line:subs", chat_id); reply("已訂閱本群。"); return
    if txt.startswith("/unsubscribe"): r.srem("line:subs", chat_id); reply("已取消訂閱。"); return
    if txt in ("/id","id"): reply(f"目前會話 ID：{chat_id}"); return
    reply("未實作的指令，請用 /summary（或 /status）")
