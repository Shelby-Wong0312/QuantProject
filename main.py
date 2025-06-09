# main.py (at project root)
import asyncio
import logging
import os
from core.event_loop import AsyncEventLoop
from data_feeds.feed_handler import AsyncMarketDataFeedHandler, MarketDataEvent

# --- 基本配置 ---
SYMBOLS = ["T.AAPL", "T.MSFT", "T.GOOG"] 
PROVIDER_URL = "wss://socket.polygon.io/stocks"
API_KEY = os.getenv("POLYGON_API_KEY") 

async def test_event_handler(event: MarketDataEvent):
    logger = logging.getLogger("EventHandler")
    logger.info(f"Received event: {event.symbol} at {event.timestamp}")

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    event_loop_instance = AsyncEventLoop()
    event_loop_instance.register_handler("MarketDataEvent", test_event_handler)

    feed_handler = AsyncMarketDataFeedHandler(
        symbols=SYMBOLS,
        api_key=API_KEY,
        event_queue=event_loop_instance.event_queue,
        provider_url=PROVIDER_URL
    )

    # --- 優雅停機函數 ---
    async def shutdown():
        logger.info("Shutdown process started...")
        await feed_handler.stop()
        await event_loop_instance.stop()
        
        # 清理其他可能的異步任務
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            [task.cancel() for task in tasks]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Graceful shutdown complete.")

    # --- 主執行邏輯 ---
    try:
        logger.info("Starting application components...")
        feed_handler.start()
        await event_loop_instance.run() 
    except asyncio.CancelledError:
        # 當 asyncio.run 被中斷時，這裡會被觸發
        pass
    finally:
        # 無論是正常結束還是被中斷，都執行停機程序
        await shutdown()

if __name__ == "__main__":
    if not API_KEY:
        print("錯誤：尚未設定 POLYGON_API_KEY 環境變數。")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            # 當使用者按下 Ctrl+C，asyncio.run() 會被中斷，
            # 進而觸發 main() 協程中的 finally 區塊。
            # 我們可以在此處印出一個更直接的訊息。
            print("\nApplication interrupted by user. Shutting down...")