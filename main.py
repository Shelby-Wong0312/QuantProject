# main.py (at project root)
import asyncio
import logging
import os
from core.event import MarketDataEvent, SignalEvent, OrderEvent, OrderType, SignalAction
from data_feeds.feed_handler import AsyncMarketDataFeedHandler
from strategy.strategy_manager import StrategyManager
from risk_management.async_risk_manager import AsyncRiskManager

# --- Alpaca 配置 ---
PROVIDER_URL = "wss://stream.data.alpaca.markets/v2/iex" 
SYMBOLS_TO_TRADE = ["AAPL", "MSFT", "GOOG", "SPY"] 
STRATEGY_CONFIGS = {
    "AAPL": {"strategy_type": "StatefulStrategy"},
    "MSFT": {"strategy_type": "StatefulStrategy"},
    "GOOG": {"strategy_type": "StatefulStrategy"},
    "SPY":  {"strategy_type": "StatefulStrategy"},
}
API_KEY_ID = os.getenv("APCA_API_KEY_ID")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    market_event_queue = asyncio.Queue()
    signal_event_queue = asyncio.Queue()
    order_event_queue = asyncio.Queue()

    # --- 實例化組件 (使用 Alpaca 金鑰) ---
    feed_handler = AsyncMarketDataFeedHandler(
        symbols=SYMBOLS_TO_TRADE,
        api_key_id=API_KEY_ID,
        secret_key=API_SECRET_KEY,
        event_queue=market_event_queue,
        provider_url=PROVIDER_URL
    )
    strategy_manager = StrategyManager(
        event_queue_in=market_event_queue,
        event_queue_out=signal_event_queue,
        strategy_configs=STRATEGY_CONFIGS
    )
    risk_manager = AsyncRiskManager(
        event_queue_in=signal_event_queue,
        event_queue_out=order_event_queue
    )

    running = True
    async def main_loop():
        while running:
            try:
                order = await asyncio.wait_for(order_event_queue.get(), timeout=1.0)
                logger.info(f"✅ MAIN LOOP GOT FINAL ORDER: {order}")
                order_event_queue.task_done()
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
        await risk_manager.stop()
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
        risk_manager.start()
        await main_loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await shutdown()

if __name__ == "__main__":
    if not API_KEY_ID or not API_SECRET_KEY:
        print("錯誤：尚未設定 APCA_API_KEY_ID 和/或 APCA_API_SECRET_KEY 環境變數。")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nApplication interrupted by user. Shutting down...")