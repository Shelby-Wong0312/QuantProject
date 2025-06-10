# data_feeds/feed_handler.py

import asyncio
import logging
from typing import List

from alpaca.data.live.stock import StockDataStream, DataFeed
from alpaca.data.models import Trade, Quote

from core.event_types import MarketDataEvent
from core import config
from core import utils

logger = logging.getLogger(__name__)

class FeedHandler:
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
        self._feed = DataFeed.IEX 
        self.stream = StockDataStream(
            api_key=self._api_key, secret_key=self._secret_key,
            feed=self._feed, url_override=config.ALPACA_DATA_URL,
            raw_data=False
        )
        self._running = False
        self._connection_attempts = 0
        self._max_connection_attempts = 5

    async def _on_trade(self, trade_data: Trade):
        try:
            event_timestamp = trade_data.timestamp.astimezone(utils.get_current_timestamp().tzinfo) \
                if trade_data.timestamp else utils.get_current_timestamp()
            event = MarketDataEvent(
                timestamp=event_timestamp, symbol=trade_data.symbol,
                data_type="TRADE", price=float(trade_data.price),
                volume=int(trade_data.size)
            )
            await self.market_data_queue.put(event)
            logger.debug(f"TRADE Event: {event}")
        except Exception as e:
            logger.error(f"Error processing trade data: {trade_data}. Error: {e}", exc_info=True)

    async def _on_quote(self, quote_data: Quote):
        try:
            event_timestamp = quote_data.timestamp.astimezone(utils.get_current_timestamp().tzinfo) \
                if quote_data.timestamp else utils.get_current_timestamp()
            event = MarketDataEvent(
                timestamp=event_timestamp, symbol=quote_data.symbol,
                data_type="QUOTE", bid_price=float(quote_data.bid_price),
                ask_price=float(quote_data.ask_price), bid_size=int(quote_data.bid_size),
                ask_size=int(quote_data.ask_size)
            )
            await self.market_data_queue.put(event)
            logger.debug(f"QUOTE Event: {event}")
        except Exception as e:
            logger.error(f"Error processing quote data: {quote_data}. Error: {e}", exc_info=True)

    async def _connect_and_subscribe(self):
        logger.info("FeedHandler: Attempting to connect to Alpaca market data stream...")
        try:
            self.stream.subscribe_trades(self._on_trade, *self.symbols)
            self.stream.subscribe_quotes(self._on_quote, *self.symbols)
            logger.info(f"FeedHandler: Subscribed to trades and quotes for symbols: {self.symbols}")
            self._connection_attempts = 0
        except Exception as e:
            logger.error(f"FeedHandler: Error during subscription: {e}", exc_info=True)
            raise

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("FeedHandler starting...")
        loop = asyncio.get_running_loop()
        while self._running and not shutdown_event.is_set():
            try:
                await self._connect_and_subscribe()
                # 使用 run_in_executor 在背景執行緒中運行阻塞的 run() 函式
                await loop.run_in_executor(None, self.stream.run)
                
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
            wait_time = min(2 ** self._connection_attempts, 60)
            logger.info(f"FeedHandler: Retrying connection in {wait_time} seconds...")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
                if shutdown_event.is_set():
                    break
            except asyncio.TimeoutError:
                pass
        await self.stop()
        logger.info("FeedHandler stopped.")

    async def stop(self):
        self._running = False
        logger.info("FeedHandler: Stopping stream...")
        try:
            # stop() 方法本身不是異步的，但關閉 stream 是異步的
            self.stream.stop()
            await self.stream.close()
            logger.info("FeedHandler: Alpaca market data stream closed.")
        except Exception as e:
            logger.error(f"FeedHandler: Error during stream stop: {e}", exc_info=True)