# 檔案位置: live_trading_app/multi_stock_main_live.py

import logging
import threading
import asyncio
from queue import Queue
from typing import List, Dict

# 導入我們建立的所有模組
from core.event_loop import AsyncEventLoop
from data_feeds.historical_feed_handler import HistoricalDataFeedHandler
from execution.capital_client import AsyncCapitalComClient
from live_trading_app.simple_execution_handler import SimpleExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy
from live_trading_app.simple_portfolio_manager import SimplePortfolioManager

def load_popular_stocks(filepath: str = "popular_stocks.txt", limit: int = 100) -> List[str]:
    """載入流行股票列表"""
    stocks = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                stock = line.strip()
                if stock and not stock.startswith('#'):
                    stocks.append(stock)
                if len(stocks) >= limit:
                    break
    except FileNotFoundError:
        logging.error(f"找不到文件: {filepath}")
        # 使用預設的熱門股票
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    
    return stocks

class MultiStockDataFeedHandler:
    """處理多支股票的數據源"""
    def __init__(self, event_queue: asyncio.Queue, symbols: List[str]):
        self.event_queue = event_queue
        self.symbols = symbols
        self._running = False
        
    def start_feed(self):
        """開始模擬數據流"""
        import asyncio
        from datetime import datetime, timezone
        from core.event_types import MarketDataEvent
        import random
        
        # 創建新的事件循環
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _async_feed():
            self._running = True
            logging.info(f"開始模擬 {len(self.symbols)} 支股票的數據流")
            
            while self._running:
                for symbol in self.symbols:
                    if not self._running:
                        break
                    
                    # 模擬市場數據
                    base_price = random.uniform(50, 500)
                    market_event = MarketDataEvent(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        event_type="LIVE_QUOTE",
                        data={
                            "open": base_price,
                            "high": base_price * 1.02,
                            "low": base_price * 0.98,
                            "close": base_price * random.uniform(0.99, 1.01),
                            "volume": random.randint(1000000, 10000000)
                        }
                    )
                    
                    self.event_queue.put(market_event)
                    await asyncio.sleep(0.1)  # 每個股票間隔0.1秒
                
                await asyncio.sleep(1)  # 每輪間隔1秒
        
        try:
            loop.run_until_complete(_async_feed())
        finally:
            loop.close()
    
    def stop(self):
        self._running = False

async def main():
    # --- 1. 設定基礎設施 ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    event_loop = AsyncEventLoop()

    # --- 2. 載入股票列表 ---
    stocks = load_popular_stocks("popular_stocks.txt", limit=10)  # 先從10支開始測試
    logger.info(f"將監控以下股票: {stocks}")

    # --- 3. 初始化系統組件 ---
    try:
        capital_client = AsyncCapitalComClient()
    except Exception as e:
        logger.error(f"初始化 CapitalComClient 失敗: {e}")
        return

    # --- 4. 創建數據源 ---
    feed_handler = MultiStockDataFeedHandler(
        event_queue=event_loop.event_queue,
        symbols=stocks
    )
    
    # --- 5. 創建執行處理器和投資組合管理器 ---
    exec_handler = SimpleExecutionHandler(
        event_queue=event_loop.event_queue,
        capital_client=capital_client
    )
    
    portfolio_manager = SimplePortfolioManager(
        event_queue=event_loop.event_queue,
        initial_cash=100000.0
    )

    # --- 6. 為每支股票創建策略和適配器 ---
    strategies = {}
    adapters = {}
    
    for symbol in stocks:
        # 每支股票使用獨立的策略實例
        strategy_params = {
            'symbol': symbol,
            'short_ma_period': 5,
            'long_ma_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'live_trade_quantity': 1  # 每次交易1股
        }
        
        strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)
        strategies[symbol] = strategy
        
        # 創建適配器
        adapter = LiveTradingAdapter(
            event_queue=event_loop.event_queue,
            abstract_strategy=strategy
        )
        adapters[symbol] = adapter

    # --- 7. 註冊事件處理器 ---
    # 創建路由器來分發市場數據到對應的適配器
    async def route_market_data(event):
        """路由市場數據到對應的適配器"""
        symbol = event.symbol
        if symbol in adapters:
            await adapters[symbol].handle_market_data_event(event)
    
    event_loop.register_handler("MarketDataEvent", route_market_data)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    # --- 8. 啟動系統 ---
    feed_thread = threading.Thread(target=feed_handler.start_feed, daemon=True)
    feed_thread.start()

    logger.info("多股票交易系統啟動...")
    
    try:
        await event_loop.run()
    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉...")
        feed_handler.stop()
    except Exception as e:
        logger.error(f"系統錯誤: {e}", exc_info=True)
    finally:
        logger.info("交易系統已關閉")

if __name__ == "__main__":
    asyncio.run(main()) 