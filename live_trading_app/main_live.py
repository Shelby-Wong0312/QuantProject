# 檔案位置: live_trading_app/main_live.py

import logging
import threading
from queue import Queue

# 導入我們建立的所有模組
from core.event_loop import EventLoop
from data_feeds.feed_handler import HistoricalDataFeedHandler
from execution.capital_client import CapitalComClient
from execution.execution_handler import ExecutionHandler
from adapters.live_trading_adapter import LiveTradingAdapter
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy
from portfolio.portfolio_manager import PortfolioManager # --- 新增導入 PortfolioManager ---

def main():
    # --- 1. 設定基礎設施 ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    event_loop = EventLoop()

    # --- 2. 初始化所有系統組件 ---
    try:
        capital_client = CapitalComClient()
    except ValueError as e:
        print(f"初始化 CapitalComClient 失敗: {e}")
        return

    strategy_params = {
        'symbol': 'BTCUSD', 'short_ma_period': 10, 'long_ma_period': 30,
        'rsi_period': 14, 'rsi_oversold': 30, 'live_trade_quantity': 0.01
    }
    strategy = AbstractEnhancedRsiMaKdStrategy(parameters=strategy_params)

    feed_handler = HistoricalDataFeedHandler(
        event_queue=event_loop.event_queue, csv_filepath='btc_usd_daily.csv',
        symbol='BTCUSD', interval_seconds=0
    )
    live_adapter = LiveTradingAdapter(
        event_queue=event_loop.event_queue, abstract_strategy=strategy
    )
    exec_handler = ExecutionHandler(
        event_queue=event_loop.event_queue, capital_client=capital_client
    )
    
    # --- 新增 PortfolioManager 的實例化 ---
    portfolio_manager = PortfolioManager(
        event_queue=event_loop.event_queue, 
        initial_cash=100000.0
    )

    # --- 3. 註冊事件與處理器的對應關係 ---
    event_loop.register_handler("MarketDataEvent", live_adapter.handle_market_data_event)
    event_loop.register_handler("SignalEvent", exec_handler.handle_signal_event)
    # --- 新增 FillEvent 的處理器註冊 ---
    event_loop.register_handler("FillEvent", portfolio_manager.handle_fill_event)
    
    # --- 4. 啟動系統 ---
    feed_thread = threading.Thread(target=feed_handler.start_feed, daemon=True)
    feed_thread.start()

    event_loop.start()


if __name__ == "__main__":
    main()
    