import os, sys
from datetime import datetime, timezone, timedelta
from linebot import LineBotApi
from linebot.models import TextSendMessage

token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
target = os.getenv("LINE_TARGET_ID")  # userId 或 groupId

if not token or not target:
    print("LINE_CHANNEL_ACCESS_TOKEN 或 LINE_TARGET_ID 未設定")
    sys.exit(1)

now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
msg = f"QuantProject 心跳測試 {now}"
LineBotApi(token).push_message(target, TextSendMessage(text=msg))
print("已送出測試訊息：", now)
