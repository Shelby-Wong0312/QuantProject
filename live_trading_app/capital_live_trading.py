# live_trading_app/capital_live.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from typing import List

from core.event_loop import AsyncEventLoop
from data_feeds.capital_live_feed import CapitalLiveFeedHandler
from live_trading_app.capital_execution_handler import CapitalExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
# vvvvvv 導入我們新的 LevelOneStrategy vvvvvv
from strategy.concrete_strategies.level_one_strategy import LevelOneStrategy
# ^^^^^^ 導入我們新的 LevelOneStrategy ^^^^^^
from live_trading_app.simple_portfolio_manager import SimplePortfolioManager

logger = logging.getLogger(__name__)

def load_symbols(filepath: str = "valid_tickers.txt", limit: int = None) -> List[str]:
    symbols = []
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        if limit:
            symbols = symbols[:limit]
    except FileNotFoundError:
        logger.error(f"找不到文件: {filepath}")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    return symbols

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    required_env_vars = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_API_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"缺少必要的环境变量: {missing_vars}")
        return
    
    symbols = load_symbols("valid_tickers.txt")
    logger.info(f"將監控以下股票: {symbols}")
    
    event_loop = AsyncEventLoop()
    feed_handler = CapitalLiveFeedHandler(event_queue=event_loop.event_queue, symbols=symbols)
    exec_handler = CapitalExecutionHandler(event_queue=event_loop.event_queue)
    portfolio_manager = SimplePortfolioManager(event_queue=event_loop.event_queue, initial_cash=10000.0)
    
    strategies = {}
    adapters = {}
    
    for symbol in symbols:
        strategy_params = {
            'symbol': symbol,
            # 這裡可以放入所有 LevelOneStrategy 需要的參數，
            # 如果不放，則會使用策略檔案中定義的預設值。
            'live_trade_quantity': 0.1
        }
        # vvvvvv 實例化新的 LevelOneStrategy vvvvvv
        strategy = LevelOneStrategy(parameters=strategy_params)
        # ^^^^^^ 實例化新的 LevelOneStrategy ^^^^^^
        strategies[symbol] = strategy
        
        adapter = LiveTradingAdapter(event_queue=event_loop.event_queue, abstract_strategy=strategy)
        adapters[symbol] = adapter
    
    async def route_market_data(event):
        symbol = event.symbol
        if symbol in adapters:
            await adapters[symbol].handle_market_data_event(event)
    
    event_loop.register_handler("MarketDataEvent", route_market_data)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    feed_task = asyncio.create_task(feed_handler.start_feed())
    logger.info("開始實時交易系統...")
    
    try:
        await asyncio.gather(event_loop.run(), feed_task)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        feed_handler.stop()
        await event_loop.stop()
    except Exception as e:
        logger.error(f"系統错误: {e}", exc_info=True)
    finally:
        logger.info("實時交易系統已关闭")

if __name__ == "__main__":
    asyncio.run(main())