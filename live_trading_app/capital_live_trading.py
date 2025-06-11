import logging
import asyncio
import os
from typing import List

from core.event_loop import AsyncEventLoop
from data_feeds.capital_live_feed import CapitalLiveFeedHandler
from live_trading_app.capital_execution_handler import CapitalExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy
from live_trading_app.simple_portfolio_manager import SimplePortfolioManager

def load_symbols(filepath: str = "tickers.txt", limit: int = 10) -> List[str]:
    """从文件加载股票代码"""
    symbols = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                symbol = line.strip()
                if symbol:
                    symbols.append(symbol)
                if len(symbols) >= limit:
                    break
    except FileNotFoundError:
        logger.error(f"找不到文件: {filepath}")
        # 使用默认的股票列表
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    return symbols

async def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 检查环境变量
    required_env_vars = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_API_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"缺少必要的环境变量: {missing_vars}")
        logger.error("请在 .env 文件中设置这些变量")
        return
    
    # 加载股票列表
    symbols = load_symbols("tickers.txt", limit=10)  # 先从前10个股票开始
    logger.info(f"将监控以下股票: {symbols}")
    
    # 创建事件循环
    event_loop = AsyncEventLoop()
    
    # 创建实时数据源
    feed_handler = CapitalLiveFeedHandler(
        event_queue=event_loop.event_queue,
        symbols=symbols
    )
    
    # 创建执行处理器
    exec_handler = CapitalExecutionHandler(
        event_queue=event_loop.event_queue
    )
    
    # 创建投资组合管理器
    portfolio_manager = SimplePortfolioManager(
        event_queue=event_loop.event_queue,
        initial_cash=10000.0  # 初始资金
    )
    
    # 为每个股票创建策略和适配器
    strategies = {}
    adapters = {}
    
    for symbol in symbols:
        # 每个股票使用独立的策略实例
        strategy_params = {
            'symbol': symbol,
            'short_ma_period': 10,
            'long_ma_period': 30,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'live_trade_quantity': 0.1  # 每次交易0.1股
        }
        
        strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)
        strategies[symbol] = strategy
        
        # 创建适配器
        adapter = LiveTradingAdapter(
            event_queue=event_loop.event_queue,
            abstract_strategy=strategy
        )
        adapters[symbol] = adapter
    
    # 注册事件处理器
    # 由于有多个适配器，我们需要一个路由器来分发市场数据事件
    async def route_market_data(event):
        """路由市场数据到对应的适配器"""
        symbol = event.symbol
        if symbol in adapters:
            await adapters[symbol].handle_market_data_event(event)
    
    event_loop.register_handler("MarketDataEvent", route_market_data)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    # 启动数据源
    feed_task = asyncio.create_task(feed_handler.start_feed())
    
    # 启动事件循环
    logger.info("开始实时交易系统...")
    
    try:
        await asyncio.gather(
            event_loop.run(),
            feed_task
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        feed_handler.stop()
        await event_loop.stop()
    except Exception as e:
        logger.error(f"系统错误: {e}", exc_info=True)
    finally:
        logger.info("实时交易系统已关闭")

if __name__ == "__main__":
    asyncio.run(main()) 