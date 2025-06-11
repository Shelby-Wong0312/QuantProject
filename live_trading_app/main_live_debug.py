# 調試版實時交易程序
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from dotenv import load_dotenv
from collections import deque
import pandas as pd
from datetime import datetime, timezone

# 導入必要的模組
from core.event_loop import AsyncEventLoop
from data_feeds.capital_live_feed import CapitalLiveFeedHandler
from live_trading_app.capital_execution_handler import CapitalExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy
from live_trading_app.simple_portfolio_manager import SimplePortfolioManager
from core.event_types import MarketDataEvent

# 加载环境变量
load_dotenv()

# 創建一個調試適配器來顯示策略狀態
class DebugLiveTradingAdapter(LiveTradingAdapter):
    def __init__(self, event_queue, abstract_strategy):
        super().__init__(event_queue, abstract_strategy)
        self.price_history = deque(maxlen=100)  # 保存最近100個價格
        self.tick_count = 0
        
    async def handle_market_data_event(self, event: MarketDataEvent):
        """處理市場數據事件並顯示調試信息"""
        self.tick_count += 1
        
        # 提取價格數據
        data = event.data
        price = data.get('bid', 0)
        
        if price > 0:
            self.price_history.append({
                'timestamp': event.timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1000000  # 模擬成交量
            })
        
        # 每10個tick顯示一次狀態
        if self.tick_count % 10 == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"📊 已接收 {self.tick_count} 個價格更新，歷史數據長度: {len(self.price_history)}")
            logger.info(f"💰 當前價格: ${price:,.2f}")
            
            # 如果有足夠的數據，顯示指標狀態
            if len(self.price_history) >= 30:
                df = pd.DataFrame(list(self.price_history))
                df.set_index('timestamp', inplace=True)
                
                # 計算簡單指標
                if len(df) >= 14:
                    rsi = self._calculate_rsi(df['close'], 14)
                    logger.info(f"📈 RSI(14): {rsi:.2f}")
                
                if len(df) >= 30:
                    sma_10 = df['close'].rolling(10).mean().iloc[-1]
                    sma_30 = df['close'].rolling(30).mean().iloc[-1]
                    logger.info(f"📊 MA(10): ${sma_10:,.2f}, MA(30): ${sma_30:,.2f}")
        
        # 調用父類方法
        await super().handle_market_data_event(event)
    
    def _calculate_rsi(self, prices, period):
        """簡單的RSI計算"""
        deltas = prices.diff()
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

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
        return

    # 使用新的 API Key
    os.environ["CAPITAL_API_KEY"] = "oVGhAub8ezuC9Zo1"
    
    logger.info("🚀 啟動調試版實時交易系統")
    logger.info(f"API Key: {os.getenv('CAPITAL_API_KEY')[:10]}...")

    # 創建事件循環
    event_loop = AsyncEventLoop()

    # 設置策略參數 - 更激進的參數以便更容易觸發
    strategy_params = {
        'symbol': 'BTCUSD', 
        'short_ma_period': 5,   # 更短的MA週期
        'long_ma_period': 15,   # 更短的MA週期
        'rsi_period': 14, 
        'rsi_oversold': 40,     # 更高的超賣閾值
        'rsi_overbought': 60,   # 更低的超買閾值
        'live_trade_quantity': 0.01  # 小量交易
    }
    
    # 創建策略實例
    strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)
    logger.info("✅ 策略已創建（使用更激進的參數）")

    # 創建實時數據源
    feed_handler = CapitalLiveFeedHandler(
        event_queue=event_loop.event_queue,
        symbols=['BTCUSD']
    )
    
    # 使用調試適配器
    live_adapter = DebugLiveTradingAdapter(
        event_queue=event_loop.event_queue, 
        abstract_strategy=strategy
    )
    
    # 創建執行處理器
    exec_handler = CapitalExecutionHandler(
        event_queue=event_loop.event_queue
    )
    
    # 創建投資組合管理器
    portfolio_manager = SimplePortfolioManager(
        event_queue=event_loop.event_queue, 
        initial_cash=100000.0
    )

    # 註冊事件處理器
    event_loop.register_handler("MarketDataEvent", live_adapter.handle_market_data_event)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    logger.info("📡 開始接收實時數據...")
    logger.info("⏳ 需要累積足夠的歷史數據（約30個數據點）才能開始交易")
    
    try:
        # 啟動數據源
        feed_task = asyncio.create_task(feed_handler.start_feed())
        
        # 啟動事件循環
        event_task = asyncio.create_task(event_loop.run())
        
        # 等待所有任務
        await asyncio.gather(feed_task, event_task)
        
    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉...")
        feed_handler.stop()
        await event_loop.stop()
    except Exception as e:
        logger.error(f"系統錯誤: {e}", exc_info=True)
    finally:
        logger.info("系統已關閉")

if __name__ == "__main__":
    asyncio.run(main()) 