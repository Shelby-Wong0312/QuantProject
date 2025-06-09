# main.py (at project root)
import asyncio
import logging
import os
from core.event import FillEvent # 僅用於類型提示
from data_feeds.feed_handler import AsyncMarketDataFeedHandler
from strategy.strategy_manager import StrategyManager
from risk_management.async_risk_manager import AsyncRiskManager
from portfolio.async_portfolio_manager import AsyncPortfolioManager

# --- 配置 ---
PROVIDER_URL = "wss://stream.data.alpaca.markets/v2/iex" 
SYMBOLS_TO_TRADE = ["SPY"] 
STRATEGY_CONFIGS = { "SPY":  {"strategy_type": "StatefulStrategy"} }
API_KEY_ID = os.getenv("APCA_API_KEY_ID")
API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # --- 建立所有事件隊列 ---
    market_queue, signal_queue, order_queue, fill_queue = (asyncio.Queue() for _ in range(4))

    # --- 實例化所有組件 ---
    feed_handler = AsyncMarketDataFeedHandler(symbols=SYMBOLS_TO_TRADE, api_key_id=API_KEY_ID, secret_key=API_SECRET_KEY, event_queue=market_queue, provider_url=PROVIDER_URL)
    strategy_manager = StrategyManager(event_queue_in=market_queue, event_queue_out=signal_queue, strategy_configs=STRATEGY_CONFIGS)
    risk_manager = AsyncRiskManager(event_queue_in=signal_queue, event_queue_out=order_queue)
    execution_handler = AsyncExecutionHandler(event_queue_in=order_queue, event_queue_out=fill_queue)
    portfolio_manager = AsyncPortfolioManager(event_queue_in=fill_queue) # 新增

    # --- 主循環，現在只為了讓程式保持運行 ---
    running = True
    async def main_loop():
        while running:
            # 讓出控制權，使其他背景任務能運行
            await asyncio.sleep(1)
    
    main_loop_task = asyncio.create_task(main_loop())

    # --- 優雅停機機制 ---
    async def shutdown():
        nonlocal running
        if not running: return
        logger.info("Shutdown process started...")
        running = False
        
        # 依相反順序關閉所有組件
        await feed_handler.stop()
        await strategy_manager.stop()
        await risk_manager.stop()
        await execution_handler.stop()
        await portfolio_manager.stop() # 新增
        
        main_loop_task.cancel()
        try: await main_loop_task
        except asyncio.CancelledError: pass
        logger.info("Graceful shutdown complete.")

    # --- 啟動所有組件 ---
    all_components = [feed_handler, strategy_manager, risk_manager, execution_handler, portfolio_manager]
    try:
        logger.info("Starting all application components...")
        for component in all_components:
            component.start()
        await main_loop_task
    except asyncio.CancelledError: pass
    finally: await shutdown()

# (if __name__ == "__main__": 區塊與上一版相同，請直接複製過來)
if __name__ == "__main__":
    if not API_KEY_ID or not API_SECRET_KEY:
        print("錯誤：尚未設定 APCA_API_KEY_ID 和/或 APCA_API_SECRET_KEY 環境變數。")
    else:
        try: asyncio.run(main())
        except KeyboardInterrupt: print("\nApplication interrupted by user. Shutting down...")