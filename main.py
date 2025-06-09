# main.py (at project root)
import asyncio
import logging
import os
from core.event import MarketDataEvent, SignalEvent
from data_feeds.feed_handler import AsyncMarketDataFeedHandler
from strategy.strategy_manager import StrategyManager

# --- 基本配置 ---
# 我們加入了 T.SPY，這是一個交易非常活躍的 ETF
SYMBOLS_TO_TRADE = ["T.AAPL", "T.MSFT", "T.GOOG", "T.SPY"] 
STRATEGY_CONFIGS = {
    "T.AAPL": {"strategy_type": "StatefulStrategy"},
    "T.MSFT": {"strategy_type": "StatefulStrategy"},
    "T.GOOG": {"strategy_type": "StatefulStrategy"},
    "T.SPY":  {"strategy_type": "StatefulStrategy"}, # 為 SPY 也設定策略
}
PROVIDER_URL = "wss://socket.polygon.io/stocks"
API_KEY = os.getenv("POLYGON_API_KEY") 

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    market_event_queue = asyncio.Queue()
    signal_event_queue = asyncio.Queue()

    feed_handler = AsyncMarketDataFeedHandler(
        symbols=SYMBOLS_TO_TRADE,
        api_key=API_KEY,
        event_queue=market_event_queue,
        provider_url=PROVIDER_URL
    )
    strategy_manager = StrategyManager(
        event_queue_in=market_event_queue,
        event_queue_out=signal_event_queue,
        strategy_configs=STRATEGY_CONFIGS
    )

    running = True
    async def main_loop():
        while running:
            try:
                signal = await asyncio.wait_for(signal_event_queue.get(), timeout=1.0)
                logger.info(f"MAIN LOOP GOT SIGNAL: {signal}")
                signal_event_queue.task_done()
            except asyncio.TimeoutError:
                continue
    
    main_loop_task = asyncio.create_task(main_loop())

    async def shutdown():
        nonlocal running
        if not running: return
        logger.info("Shutdown process started...")
        running = False
        await feed_handler.stop()
        await strategy_manager.stop()
        main_loop_task.cancel()
        try:
            await main_loop_task
        except asyncio.CancelledError:
            pass
        logger.info("Graceful shutdown complete.")

    try:
        logger.info("Starting application components...")
        feed_handler.start()
        strategy_manager.start()
        await main_loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await shutdown()

if __name__ == "__main__":
    if not API_KEY:
        print("錯誤：尚未設定 POLYGON_API_KEY 環境變數。")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nApplication interrupted by user. Shutting down...")