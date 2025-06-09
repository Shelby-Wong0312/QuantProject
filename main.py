# main.py (at project root)
import asyncio
import logging
import signal
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

    loop = asyncio.get_running_loop()
    event_loop_instance = AsyncEventLoop()

    event_loop_instance.register_handler("MarketDataEvent", test_event_handler)

    feed_handler = AsyncMarketDataFeedHandler(
        symbols=SYMBOLS,
        api_key=API_KEY,
        event_queue=event_loop_instance.event_queue,
        provider_url=PROVIDER_URL
    )

    # --- 設定優雅停機機制 ---
    shutdown_signals = (signal.SIGINT, signal.SIGTERM)
    async def shutdown(sig: signal.Signals):
        logger.info(f"Received exit signal {sig.name}...")
        await feed_handler.stop()
        await event_loop_instance.stop()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Graceful shutdown complete.")

    for s in shutdown_signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s)))

    # --- 啟動組件 ---
    try:
        logger.info("Starting application components...")
        feed_handler.start()
        await event_loop_instance.run() 
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        logger.info("Application has been shut down.")

if __name__ == "__main__":
    if not API_KEY:
        print("錯誤：尚未設定 POLYGON_API_KEY 環境變數。")
    else:
        asyncio.run(main())