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

logger = logging.getLogger(__name__)

def load_symbols(filepath: str = "valid_tickers.txt", limit: int = None) -> List[str]:
    """從文件加載股票代碼，監控全部股票"""
    symbols = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                symbol = line.strip()
                if symbol:
                    symbols.append(symbol)
                # 不再限制數量，監控全部股票
    except FileNotFoundError:
        logger.error(f"找不到文件: {filepath}")
        # 使用默認的股票列表
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    return symbols

async def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查环境变量
    required_env_vars = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_API_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"缺少必要的环境变量: {missing_vars}")
        logger.error("请在 .env 文件中设置这些变量")
        return
    
    # 加載股票列表
    symbols = load_symbols("valid_tickers.txt")  # 監控全部股票
    logger.info(f"將監控以下股票: {symbols}")
    
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
    
    # 為每個股票創建策略和適配器
    strategies = {}
    adapters = {}
    
    for symbol in symbols:
        # 每個股票使用獨立的策略實例
        strategy_params = {
            'symbol': symbol,
            'short_ma_period': 5,  # 統一短線MA為5
            'long_ma_period': 20, # 統一長線MA為20
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'live_trade_quantity': 0.1  # 每次交易0.1股
        }
        strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)
        strategies[symbol] = strategy
        
        # 創建適配器
        adapter = LiveTradingAdapter(
            event_queue=event_loop.event_queue,
            abstract_strategy=strategy
        )
        adapters[symbol] = adapter
    
    # 注册事件处理器
    # 由於有多个適配器，我们需要一个路由器來分發市場數據事件
    async def route_market_data(event):
        """路由市場數據到对应的適配器"""
        symbol = event.symbol
        if symbol in adapters:
            await adapters[symbol].handle_market_data_event(event)
    
    event_loop.register_handler("MarketDataEvent", route_market_data)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    # 啟動數據源
    feed_task = asyncio.create_task(feed_handler.start_feed())
    
    # 啟動事件循環
    logger.info("開始實時交易系統...")
    
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
        logger.error(f"系統错误: {e}", exc_info=True)
    finally:
        logger.info("實時交易系統已关闭")

if __name__ == "__main__":
    asyncio.run(main()) 