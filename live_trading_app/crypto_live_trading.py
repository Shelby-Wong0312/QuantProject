# 檔案位置: live_trading_app/crypto_live_trading.py

import logging
import asyncio
import os
from typing import List

from core.event_loop import AsyncEventLoop
from data_feeds.capital_live_feed import CapitalLiveFeedHandler
from live_trading_app.capital_execution_handler import CapitalExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
from strategy.concrete_strategies.crypto_level1_strategy import CryptoLevel1Strategy
from live_trading_app.simple_portfolio_manager import SimplePortfolioManager

def load_crypto_symbols(filepath: str = "popular_crypto.txt", limit: int = 10) -> List[str]:
    """從文件加載虛擬貨幣代碼"""
    symbols = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                symbol = line.strip()
                if symbol and not symbol.startswith('#'):
                    symbols.append(symbol)
                if len(symbols) >= limit:
                    break
    except FileNotFoundError:
        logger.error(f"找不到文件: {filepath}")
        # 使用默認的虛擬貨幣列表
        symbols = ["BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD", "ADAUSD"]
    
    return symbols

async def main():
    # 設置日志
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
    
    # 加載虛擬貨幣列表 - 使用確定存在的虛擬貨幣
    # 根據測試結果，這些是 Capital.com 支持的虛擬貨幣
    symbols = ["BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "XRPUSD"]  # 使用確定存在的
    logger.info(f"將監控以下虛擬貨幣: {symbols}")
    
    # 創建事件循環
    event_loop = AsyncEventLoop()
    
    # 修改 CapitalLiveFeedHandler 的 EPIC 映射
    class CryptoCapitalLiveFeedHandler(CapitalLiveFeedHandler):
        def _map_symbol_to_epic(self, symbol: str) -> str:
            """將虛擬貨幣代碼映射到 Capital.com 的 EPIC"""
            # Capital.com 直接使用標準格式，不需要映射
            # BTCUSD -> BTCUSD
            # ETHUSD -> ETHUSD
            # 等等
            return symbol
    
    # 創建實時數據源
    feed_handler = CryptoCapitalLiveFeedHandler(
        event_queue=event_loop.event_queue,
        symbols=symbols
    )
    
    # 創建執行處理器
    exec_handler = CapitalExecutionHandler(
        event_queue=event_loop.event_queue
    )
    
    # 創建投資組合管理器
    portfolio_manager = SimplePortfolioManager(
        event_queue=event_loop.event_queue,
        initial_cash=10000.0  # 初始資金
    )
    
    # 為每個虛擬貨幣創建策略和適配器
    strategies = {}
    adapters = {}
    
    for symbol in symbols:
        # 每個虛擬貨幣使用獨立的一級策略實例
        strategy_params = {
            'symbol': symbol,
            # MA 參數
            'short_ma_period': 5,
            'long_ma_period': 20,
            # BIAS 參數
            'bias_period': 20,
            'bias_upper': 7.0,
            'bias_lower': -8.0,
            # KD 參數
            'kd_k': 14,
            'kd_d': 3,
            'kd_smooth': 3,
            'kd_overbought': 80,
            'kd_oversold': 20,
            # MACD 參數
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            # RSI 參數
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            # Bollinger Bands 參數
            'bb_period': 20,
            'bb_std': 2.0,
            # Volume 參數
            'vol_ma_period': 20,
            'vol_multiplier': 1.5,
            # ATR 風險管理
            'atr_period': 14,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            # 交易量
            'live_trade_quantity': 0.01  # 每次交易0.01個單位
        }
        
        strategy = CryptoLevel1Strategy(parameters=strategy_params)
        strategies[symbol] = strategy
        
        # 創建適配器
        adapter = LiveTradingAdapter(
            event_queue=event_loop.event_queue,
            abstract_strategy=strategy
        )
        adapters[symbol] = adapter
    
    # 註冊事件處理器
    async def route_market_data(event):
        """路由市場數據到對應的適配器"""
        symbol = event.symbol
        if symbol in adapters:
            await adapters[symbol].handle_market_data_event(event)
    
    event_loop.register_handler("MarketDataEvent", route_market_data)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    # 啟動數據源
    feed_task = asyncio.create_task(feed_handler.start_feed())
    
    # 啟動事件循環
    logger.info("開始虛擬貨幣實時交易系統...")
    logger.info("虛擬貨幣市場24/7開放，隨時可以交易！")
    
    try:
        await asyncio.gather(
            event_loop.run(),
            feed_task
        )
    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉...")
        feed_handler.stop()
        await event_loop.stop()
    except Exception as e:
        logger.error(f"系統錯誤: {e}", exc_info=True)
    finally:
        logger.info("虛擬貨幣交易系統已關閉")

if __name__ == "__main__":
    asyncio.run(main()) 