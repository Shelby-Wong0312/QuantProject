# test_polygon.py (Corrected version without asyncio)

import os
import logging
from dotenv import load_dotenv
import time

# We will use the official polygon-io client library
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market

# --- Basic Setup ---
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
logging.basicConfig(level=logging.INFO)


# --- A simple SYNCHRONOUS message handler ---
def simple_message_handler(messages: list[WebSocketMessage]):
    """
    This function's only job is to print any message it receives from the server.
    """
    for msg in messages:
        # We add a timestamp to see when we receive messages
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received message: {msg}")


# --- Main execution function (synchronous) ---
def main():
    print("--- Polygon.io 連線測試 (最終修正版) ---")

    if not POLYGON_API_KEY:
        print("錯誤：找不到 POLYGON_API_KEY。請檢查您的 .env 檔案。")
        return

    print(f"找到 API 金鑰，開頭為: {POLYGON_API_KEY[:5]}...")
    
    # Create a WebSocketClient instance
    ws_client = WebSocketClient(
        api_key=POLYGON_API_KEY,
        market=Market.Stocks,
        feed="socket.polygon.io",
        subscriptions=["AM.SPY"]  # We subscribe to just one stock (SPY) for testing
    )
    
    print("正在嘗試連線並接收數據... (程式將會在此停留並持續監聽，請用 Ctrl+C 停止)")
    
    # Directly call the blocking run() method. It will manage its own loop.
    ws_client.run(handle_msg=simple_message_handler)

    print("連線已中斷。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n測試腳本被使用者手動停止。")
    except Exception as e:
        print(f"\n程式發生未預期的錯誤: {e}")