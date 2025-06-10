# main.py

import asyncio
import signal
import logging
from typing import List, Optional

from core import config
from core.utils import setup_logging
from core.event_types import SystemControlEvent

# 根據您的目錄結構，從對應位置導入組件
from data_feeds.feed_handler import FeedHandler
from strategy.strategy_manager import StrategyManager
from risk_management.risk_manager import RiskManager
from execution.execution_handler import ExecutionHandler
from portfolio.portfolio_manager import PortfolioManager # <- 已修正此處的引用

# 儘早設定日誌
setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 全局關閉事件
shutdown_event = asyncio.Event()

async def graceful_shutdown(s_event: asyncio.Event, sig: Optional[signal.Signals] = None):
    if sig:
        logger.info(f"Received shutdown signal: {sig.name}. Initiating graceful shutdown...")
    else:
        logger.info("Initiating graceful shutdown due to internal request or error...")
    
    s_event.set() # 發信號給所有組件開始關閉

async def main():
    logger.info("Starting Industrial Grade Trading Framework...")

    # 初始化所有異步隊列
    market_data_queue = asyncio.Queue(maxsize=config.MARKET_DATA_QUEUE_MAX_SIZE)
    signal_queue = asyncio.Queue(maxsize=config.SIGNAL_QUEUE_MAX_SIZE)
    order_queue = asyncio.Queue(maxsize=config.ORDER_QUEUE_MAX_SIZE)
    fill_queue = asyncio.Queue(maxsize=config.FILL_QUEUE_MAX_SIZE)

    # 實例化所有組件
    # 為了簡化骨架，此處我們暫時不讓 PortfolioManager 消費市場數據
    symbols_to_trade = ["*"] # 使用 "*" 訂閱 IEX 提供的所有股票數據

    feed_handler = FeedHandler(market_data_queue, symbols=symbols_to_trade)
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

    # 為每個組件的 run 方法創建一個異步任務
    tasks: List[asyncio.Task] = []
    for component in components:
        if hasattr(component, 'run') and asyncio.iscoroutinefunction(component.run):
            tasks.append(asyncio.create_task(component.run(shutdown_event), name=component.__class__.__name__))
        else:
            logger.error(f"Component {component.__class__.__name__} does not have a runnable async 'run' method.")

    if not tasks:
        logger.critical("No component tasks were created. Exiting.")
        return

    # 等待所有任務完成，或直到有關閉信號
    logger.info("All components initialized. Starting main event loop...")
    
    # 監聽 OS 關閉信號
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(graceful_shutdown(shutdown_event, s)))
        except NotImplementedError:
            logger.warning("Signal handlers not supported on this platform (e.g., Windows). Use Ctrl+C.")

    # 主迴圈，監控任務狀態
    while not shutdown_event.is_set():
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
        
        for task in done:
            if not task.cancelled() and task.exception():
                logger.critical(f"Critical error in task '{task.get_name()}'. Initiating system shutdown.", exc_info=task.exception())
                await graceful_shutdown(shutdown_event)
                break
        
        if shutdown_event.is_set():
            break

        if not pending and all(t.done() for t in tasks):
             logger.info("All component tasks have completed.")
             break

    logger.info("Shutdown initiated, waiting for tasks to finalize...")
    remaining_tasks = [t for t in tasks if not t.done()]
    if remaining_tasks:
        await asyncio.wait(remaining_tasks, timeout=10.0)

    for task in tasks:
        if not task.done():
            task.cancel()
    
    logger.info("Framework shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.critical(f"Application terminated due to an unhandled exception in main: {e}", exc_info=True)