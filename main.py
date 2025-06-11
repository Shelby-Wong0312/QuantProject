# main.py

import asyncio
import signal
import logging
from typing import List, Optional

# --- 1. 先導入所有模組 ---
from core import config
from core import utils
from data_feeds.feed_handler import FeedHandler
from strategy.strategy_manager import StrategyManager
from risk_management.risk_manager import RiskManager
from execution.execution_handler import ExecutionHandler
from portfolio.portfolio_manager import PortfolioManager

# --- 2. 模組導入完畢後，再執行函式 ---
utils.setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 全局關閉事件
shutdown_event = asyncio.Event()

async def graceful_shutdown(s_event: asyncio.Event, sig: Optional[signal.Signals] = None):
    if sig:
        logger.info(f"Received shutdown signal: {sig.name}. Initiating graceful shutdown...")
    else:
        logger.info("Initiating graceful shutdown due to internal request or error...")
    s_event.set()

async def main():
    logger.info("Starting Industrial Grade Trading Framework...")

    market_data_queue = asyncio.Queue(maxsize=config.MARKET_DATA_QUEUE_MAX_SIZE)
    signal_queue = asyncio.Queue(maxsize=config.SIGNAL_QUEUE_MAX_SIZE)
    order_queue = asyncio.Queue(maxsize=config.ORDER_QUEUE_MAX_SIZE)
    fill_queue = asyncio.Queue(maxsize=config.FILL_QUEUE_MAX_SIZE)

    # 實例化所有組件，並傳入從檔案讀取的完整股票列表
    feed_handler = FeedHandler(
        market_data_queue, 
        symbols=config.SYMBOLS_TO_TRADE, 
        api_key=config.POLYGON_API_KEY
    )

    strategy_manager = StrategyManager(market_data_queue, signal_queue)
    risk_manager = RiskManager(signal_queue, order_queue)
    execution_handler = ExecutionHandler(order_queue, fill_queue)
    portfolio_manager = PortfolioManager(fill_queue, market_data_queue=None)

    components = [
        feed_handler, 
        strategy_manager, 
        risk_manager, 
        execution_handler, 
        portfolio_manager
    ]

    tasks: List[asyncio.Task] = []
    for component in components:
        tasks.append(asyncio.create_task(component.run(shutdown_event), name=component.__class__.__name__))

    logger.info("All components initialized. Starting main event loop...")
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(graceful_shutdown(shutdown_event, s)))
        except NotImplementedError:
            logger.warning("Signal handlers not supported on this platform (e.g., Windows). Use Ctrl+C.")

    while not shutdown_event.is_set():
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
        for task in done:
            if not task.cancelled() and task.exception():
                logger.critical(f"Critical error in task '{task.get_name()}'. Initiating shutdown.", exc_info=task.exception())
                await graceful_shutdown(shutdown_event)
                break
        if not pending:
             logger.info("All component tasks have completed.")
             break

    logger.info("Shutdown initiated, waiting for tasks to finalize...")
    remaining_tasks = [t for t in tasks if not t.done()]
    if remaining_tasks:
        await asyncio.wait(remaining_tasks, timeout=10.0)
    
    logger.info("Framework shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.critical(f"Application terminated due to an unhandled exception in main: {e}", exc_info=True)