# data_feeds/feed_handler.py

import asyncio
import logging
from typing import List

from alpaca.data.live.stock import StockDataStream, DataFeed
from alpaca.data.models import Trade, Quote

from core.event_types import MarketDataEvent
from core import config
from core.utils import get_current_timestamp

logger = logging.getLogger(__name__)

class FeedHandler:
    """
    負責處理來自 Alpaca 的即時市場數據流，並將其轉換為 MarketDataEvent。
    """
    def __init__(self,
                 market_data_queue: asyncio.Queue,
                 symbols: List[str],
                 api_key: str = config.ALPACA_API_KEY_ID,
                 secret_key: str = config.ALPACA_SECRET_KEY,
                 paper: bool = config.ALPACA_PAPER_TRADING):
        self.market_data_queue = market_data_queue
        self.symbols = symbols
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._feed = DataFeed.IEX # 使用免費的 IEX 數據源

        self.stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=self._feed,
            url_override=config.ALPACA_DATA_URL,
            raw_data=False # 讓 SDK 將數據處理成 Pydantic 模型
        )
        
        self._running = False
        self._connection_attempts = 0
        self._max_connection_attempts = 5 # Example

    async def _on_trade(self, trade_data: Trade):
        try:
            event_timestamp = trade_data.timestamp.astimezone(get_current_timestamp().tzinfo) \
                if trade_data.timestamp else get_current_timestamp()

            event = MarketDataEvent(
                timestamp=event_timestamp,
                symbol=trade_data.symbol,
                data_type="TRADE",
                price=float(trade_data.price),
                volume=int(trade_data.size)
            )
            await self.market_data_queue.put(event)
            logger.debug(f"TRADE Event: {event}")
        except Exception as e:
            logger.error(f"Error processing trade data: {trade_data}. Error: {e}", exc_info=True)

    async def _on_quote(self, quote_data: Quote):
        try:
            event_timestamp = quote_data.timestamp.astimezone(get_current_timestamp().tzinfo) \
                if quote_data.timestamp else get_current_timestamp()
            
            event = MarketDataEvent(
                timestamp=event_timestamp,
                symbol=quote_data.symbol,
                data_type="QUOTE",
                bid_price=float(quote_data.bid_price),
                ask_price=float(quote_data.ask_price),
                bid_size=int(quote_data.bid_size),
                ask_size=int(quote_data.ask_size)
            )
            await self.market_data_queue.put(event)
            logger.debug(f"QUOTE Event: {event}")
        except Exception as e:
            logger.error(f"Error processing quote data: {quote_data}. Error: {e}", exc_info=True)

    async def _connect_and_subscribe(self):
        logger.info("FeedHandler: Attempting to connect to Alpaca market data stream...")
        try:
            # 訂閱交易和報價數據
            self.stream.subscribe_trades(self._on_trade, *self.symbols)
            self.stream.subscribe_quotes(self._on_quote, *self.symbols)
            logger.info(f"FeedHandler: Subscribed to trades and quotes for symbols: {self.symbols}")
            self._connection_attempts = 0 # 連接成功後重置嘗試次數
        except Exception as e:
            logger.error(f"FeedHandler: Error during subscription: {e}", exc_info=True)
            raise # 重新拋出異常，由 run 迴圈處理

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("FeedHandler starting...")

        while self._running and not shutdown_event.is_set():
            try:
                # 建立連接並運行 stream 的主事件迴圈
                # stream.run() 是一個阻塞調用，它會處理 WebSocket 連接和消息
                await self._connect_and_subscribe()
                await self.stream.run() # 此處會持續運行直到連接中斷
                
                # 如果 run() 正常退出 (例如手動停止)，檢查是否需要關閉
                if shutdown_event.is_set():
                    break
                
                logger.warning("FeedHandler: Stream run method exited unexpectedly. Reconnecting...")

            except Exception as e:
                logger.error(f"FeedHandler: An error occurred: {e}", exc_info=True)

            if shutdown_event.is_set():
                break

            self._connection_attempts += 1
            if self._connection_attempts >= self._max_connection_attempts:
                logger.critical("FeedHandler: Max connection attempts reached. Stopping.")
                break

            # 指數退避重試
            wait_time = min(2 ** self._connection_attempts, 60)
            logger.info(f"FeedHandler: Retrying connection in {wait_time} seconds...")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
                if shutdown_event.is_set():
                    break
            except asyncio.TimeoutError:
                pass # 正常等待，繼續重試

        await self.stop()
        logger.info("FeedHandler stopped.")

    async def stop(self):
        self._running = False
        logger.info("FeedHandler: Stopping stream...")
        try:
            if self.stream:
                self.stream.unsubscribe_trades(*self.symbols)
                self.stream.unsubscribe_quotes(*self.symbols)
                await self.stream.close()
                logger.info("FeedHandler: Alpaca market data stream closed.")
        except Exception as e:
            logger.error(f"FeedHandler: Error during stream stop: {e}", exc_info=True)
