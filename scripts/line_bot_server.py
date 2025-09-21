import os, json, redis
from datetime import datetime, timezone, timedelta
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
secret = os.getenv("LINE_CHANNEL_SECRET")
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

if not token or not secret:
    raise SystemExit("請在 .env 設定 LINE_CHANNEL_ACCESS_TOKEN 與 LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(token)
handler = WebhookHandler(secret)
r = redis.StrictRedis.from_url(redis_url)
SUB_KEY = "line:subs"

STATUS_JSON = "/app/capital_api_status.json"
MAX_AGE_MIN = 15

def fresh(dtstr):
    try:
        dt = datetime.fromisoformat(dtstr.replace("Z","+00:00"))
    except Exception:
        return True
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)) <= timedelta(minutes=MAX_AGE_MIN)

def get_summary_text():
    # 這裡先讀診斷 JSON（你已經在容器內每分鐘更新），之後可換成實際 Equity/Cash 的來源
    if not os.path.exists(STATUS_JSON):
        return "【系統狀態】找不到 capital_api_status.json"
    with open(STATUS_JSON,"r",encoding="utf-8") as f:
        data = json.load(f)
    ts = data.get("timestamp","(無時間戳)")
    res = data.get("results",{})
    ok = lambda k: "" if res.get(k) else ""
    lines = [
        "【系統狀態 / System Status】",
        f"Auth: {ok('authentication')}  Docs: {ok('documentation')}  WS: {ok('websocket')}",
        f"Demo: {ok('demo_api')}  Live: {ok('live_api')}",
        f"Updated: {ts}  ({'新鮮' if fresh(ts) else '可能過期'})",
        "",
        "指令：/summary  /subscribe  /unsubscribe  /id  /help",
    ]
    return "\n".join(lines)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature","")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def on_message(event):
    txt = (event.message.text or "").strip().lower()
    st = event.source.type
    if st == "group":
        chat_id = event.source.group_id
    elif st == "room":
        chat_id = event.source.room_id
    else:
        chat_id = event.source.user_id

    def reply(text):
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))

    if txt in ("/help","help"):
        reply("指令：/summary /subscribe /unsubscribe /id /help")
    elif txt in ("/id","id"):
        reply(f"目前會話 ID：{chat_id}")
    elif txt.startswith("/subscribe"):
        r.sadd(SUB_KEY, chat_id); reply("已訂閱本群，之後可定時推送。")
    elif txt.startswith("/unsubscribe"):
        r.srem(SUB_KEY, chat_id); reply("已取消訂閱。")
    elif txt.startswith("/summary"):
        reply(get_summary_text())
    else:
        # 靜默或回簡短提示
        if any(k in txt for k in ["/","summary","subscribe","unsubscribe","help","id"]):
            reply("未知指令，輸入 /help 看指令列表。")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")))
