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

    # 使用新的 API Key
    os.environ["CAPITAL_API_KEY"] = "oVGhAub8ezuC9Zo1"
    
    logger.info("環境變量檢查通過")
    logger.info(f"API Key: {os.getenv('CAPITAL_API_KEY')[:10]}...")
    logger.info(f"Identifier: {os.getenv('CAPITAL_IDENTIFIER')}")

    # 創建事件循環
    event_loop = AsyncEventLoop()

    # 設置策略參數
    strategy_params = {
        'symbol': 'BTCUSD', 
        'short_ma_period': 10, 
        'long_ma_period': 30,
        'rsi_period': 14, 
        'rsi_oversold': 30, 
        'rsi_overbought': 70,
        'live_trade_quantity': 0.1
    }
    
    # 創建策略實例
    strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)
    logger.info("策略已創建")

    # 創建實時數據源
    feed_handler = CapitalLiveFeedHandler(
        event_queue=event_loop.event_queue,
        symbols=['BTCUSD']
    )
    logger.info("數據源已創建")
    
    # 創建策略適配器
    live_adapter = LiveTradingAdapter(
        event_queue=event_loop.event_queue, 
        abstract_strategy=strategy
    )
    logger.info("策略適配器已創建")
    
    # 創建執行處理器
    exec_handler = CapitalExecutionHandler(
        event_queue=event_loop.event_queue
    )
    logger.info("執行處理器已創建")
    
    # 創建投資組合管理器
    portfolio_manager = SimplePortfolioManager(
        event_queue=event_loop.event_queue, 
        initial_cash=100000.0
    )
    logger.info("投資組合管理器已創建")

    # 註冊事件處理器
    event_loop.register_handler("MarketDataEvent", live_adapter.handle_market_data_event)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    logger.info("事件處理器已註冊")
    
    # 啟動系統
    logger.info("開始實時交易系統...")
    logger.info("監控 BTCUSD，使用 RSI + MA 策略")
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
        logger.info("實時交易系統已關閉")

if __name__ == "__main__":
    asyncio.run(main()) 