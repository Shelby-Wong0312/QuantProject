# main.py (at project root)
import asyncio
import logging
import os
from core.event import MarketDataEvent, SignalEvent, OrderEvent
from data_feeds.feed_handler import AsyncMarketDataFeedHandler
from strategy.strategy_manager import StrategyManager
from risk_management.async_risk_manager import AsyncRiskManager

# --- 基本配置 ---
# 注意：我們加入了 GOOG 來測試風險管理器的黑名單功能
SYMBOLS_TO_TRADE = ["T.AAPL", "T.MSFT", "T.GOOG", "T.SPY"] 
STRATEGY_CONFIGS = {
    "T.AAPL": {"strategy_type": "StatefulStrategy"},
    "T.MSFT": {"strategy_type": "StatefulStrategy"},
    "T.GOOG": {"strategy_type": "StatefulStrategy"}, # GOOG 會被風控攔截
    "T.SPY":  {"strategy_type": "StatefulStrategy"},
}
PROVIDER_URL = "wss://socket.polygon.io/stocks"
API_KEY = os.getenv("POLYGON_API_KEY") 

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # --- 建立三層事件隊列 ---
    market_event_queue = asyncio.Queue()
    signal_event_queue = asyncio.Queue()
    order_event_queue = asyncio.Queue() # 新增：訂單事件隊列

    # --- 實例化所有組件 ---
    feed_handler = AsyncMarketDataFeedHandler(
        symbols=SYMBOLS_TO_TRADE,
        api_key=API_KEY,
        event_queue=market_event_queue, # 輸出到市場隊列
        provider_url=PROVIDER_URL
    )
    strategy_manager = StrategyManager(
        event_queue_in=market_event_queue,  # 從市場隊列讀取
        event_queue_out=signal_event_queue, # 輸出到信號隊列
        strategy_configs=STRATEGY_CONFIGS
    )
    risk_manager = AsyncRiskManager(
        event_queue_in=signal_event_queue, # 從信號隊列讀取
        event_queue_out=order_event_queue  # 輸出到訂單隊列
    )

    # --- 主事件循環，現在監聽最終的訂單事件 ---
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

    # --- 優雅停機機制 ---
    async def shutdown():
        nonlocal running
        if not running: return
        logger.info("Shutdown process started...")
        running = False
        
        # 依照與啟動相反的順序關閉
        await feed_handler.stop()
        await strategy_manager.stop()
        await risk_manager.stop() # 新增
        
        main_loop_task.cancel()
        try:
            await main_loop_task
        except asyncio.CancelledError:
            pass
        logger.info("Graceful shutdown complete.")

    # --- 啟動所有組件 ---
    try:
        logger.info("Starting application components...")
        feed_handler.start()
        strategy_manager.start()
        risk_manager.start() # 新增
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
            