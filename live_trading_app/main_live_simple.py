# 簡化版實時交易程序
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from dotenv import load_dotenv

# 導入必要的模組
from core.event_loop import AsyncEventLoop
from data_feeds.capital_live_feed import CapitalLiveFeedHandler
from live_trading_app.capital_execution_handler import CapitalExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy
from live_trading_app.simple_portfolio_manager import SimplePortfolioManager

# 讀取 popular_stocks.txt

def load_stocks(filepath="tickers.txt", limit=None):
    stocks = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                symbol = line.strip()
                if symbol and not symbol.startswith('#'):
                    stocks.append(symbol)
    except FileNotFoundError:
        print(f"找不到 {filepath}，請確認檔案存在")
    return stocks

# 加载环境变量
load_dotenv()

async def main():
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 檢查環境變量
    required_env_vars = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_API_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"缺少必要的環境變量: {missing_vars}")
        logger.error("請在 .env 文件中設置這些變量")
        return

    # 讀取股票清單
    symbols = load_stocks()
    logger.info(f"將監控以下股票: {symbols}")

    # 使用新的 API Key
    os.environ["CAPITAL_API_KEY"] = "oVGhAub8ezuC9Zo1"
    
    logger.info("環境變量檢查通過")
    logger.info(f"API Key: {os.getenv('CAPITAL_API_KEY')[:10]}...")
    logger.info(f"Identifier: {os.getenv('CAPITAL_IDENTIFIER')}")

    # 創建事件循環
    event_loop = AsyncEventLoop()

    # 建立實時數據源
    feed_handler = CapitalLiveFeedHandler(
        event_queue=event_loop.event_queue,
        symbols=symbols
    )
    logger.info("數據源已創建")
    
    # 建立執行處理器
    exec_handler = CapitalExecutionHandler(
        event_queue=event_loop.event_queue
    )
    logger.info("執行處理器已創建")
    
    # 建立投資組合管理器
    portfolio_manager = SimplePortfolioManager(
        event_queue=event_loop.event_queue, 
        initial_cash=100000.0
    )
    logger.info("投資組合管理器已創建")

    # 為每支股票建立策略與適配器
    strategies = {}
    adapters = {}
    for symbol in symbols:
        strategy_params = {
            'symbol': symbol,
            'short_ma_period': 5,
            'long_ma_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'live_trade_quantity': 1
        }
        strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)
        strategies[symbol] = strategy
        adapter = LiveTradingAdapter(
            event_queue=event_loop.event_queue,
            abstract_strategy=strategy
        )
        adapters[symbol] = adapter

    # 註冊事件處理器
    async def route_market_data(event):
        symbol = event.symbol
        if symbol in adapters:
            await adapters[symbol].handle_market_data_event(event)

    event_loop.register_handler("MarketDataEvent", route_market_data)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    logger.info("事件處理器已註冊")
    
    # 啟動系統
    logger.info("開始多股票實時交易系統...")
    logger.info("數據更新間隔: 0秒（最快速度）")
    
    try:
        # 啟動數據源
        feed_task = asyncio.create_task(feed_handler.start_feed())
        logger.info("數據源已啟動")
        
        # 啟動事件循環
        event_task = asyncio.create_task(event_loop.run())
        logger.info("事件循環已啟動")
        
        # 等待所有任務
        await asyncio.gather(feed_task, event_task)
        
    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉...")
        feed_handler.stop()
        await event_loop.stop()
    except Exception as e:
        logger.error(f"系統錯誤: {e}", exc_info=True)
    finally:
        logger.info("多股票實時交易系統已關閉")

if __name__ == "__main__":
    asyncio.run(main()) 