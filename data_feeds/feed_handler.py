# data_feeds/feed_handler.py
import asyncio
import websockets # Using the 'websockets' library
import json
import logging
import pandas as pd # 修正：導入 pandas 函式庫
from core.event import MarketDataEvent # Assuming your event definitions

logger = logging.getLogger(__name__)

class AsyncMarketDataFeedHandler:
    def __init__(self, symbols: list, api_key: str, event_queue: asyncio.Queue, provider_url: str):
        self.symbols = symbols # List of symbols like "T.AAPL", "Q.MSFT" for Polygon.io
        self.api_key = api_key
        self.event_queue = event_queue
        self.provider_url = provider_url # e.g., "wss://socket.polygon.io/stocks"
        self._connection_tasks = []
        self._running = False
        # Max symbols per connection (example, adjust based on provider limits)
        self.MAX_SYMBOLS_PER_CONNECTION = 500

    async def _handle_connection(self, symbols_for_this_connection):
        async for websocket in websockets.connect(self.provider_url, ping_interval=20, ping_timeout=20):
            try:
                # 1. Authenticate
                auth_payload = {"action": "auth", "params": self.api_key}
                await websocket.send(json.dumps(auth_payload))
                auth_response = await websocket.recv() # Wait for auth confirmation
                logger.info(f"Auth response for {symbols_for_this_connection[:5]}...: {auth_response}")
                # TODO: Check if auth_response is successful

                # 2. Subscribe to symbols
                subscribe_payload = {"action": "subscribe", "params": ",".join(symbols_for_this_connection)}
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Subscribed to {len(symbols_for_this_connection)} symbols on one connection.")

                # 3. Receive messages
                while self._running:
                    message_str = await websocket.recv()
                    messages = json.loads(message_str) # Polygon sends array of messages
                    for msg_data in messages:
                        # TODO: Parse msg_data (trade, quote, bar) based on msg_data.get('ev')
                        # and transform into your MarketDataEvent structure
                        # Example for a trade event:
                        if msg_data.get('ev') == 'T': # Trade event for Polygon
                            market_event = MarketDataEvent(
                                symbol=msg_data.get('sym'),
                                timestamp=pd.to_datetime(msg_data.get('t'), unit='ms'), # 此行現在可以正確執行
                                # ... populate other fields like price, volume from msg_data
                                # ohlcv_data = pd.DataFrame(...) or pd.Series(...)
                            )
                            await self.event_queue.put(market_event)

            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"Connection closed for symbols {symbols_for_this_connection[:5]}...: {e}. Reconnecting...")
                await asyncio.sleep(5) # Wait before retrying
                continue # Reconnects due to the outer async for loop
            except Exception as e:
                logger.error(f"Error handling connection for {symbols_for_this_connection[:5]}...: {e}", exc_info=True)
                await asyncio.sleep(15) # Longer wait for unexpected errors
                # Potentially break or implement more sophisticated error handling/reconnect limits

    def start(self):
        if not self._running:
            self._running = True
            # Split symbols among multiple connections
            for i in range(0, len(self.symbols), self.MAX_SYMBOLS_PER_CONNECTION):
                symbols_chunk = self.symbols[i:i + self.MAX_SYMBOLS_PER_CONNECTION]
                task = asyncio.create_task(self._handle_connection(symbols_chunk))
                self._connection_tasks.append(task)
            logger.info(f"Market Data Feed Handler started with {len(self._connection_tasks)} connection tasks.")

    async def stop(self):
        self._running = False
        logger.info("Stopping Market Data Feed Handler...")
        for task in self._connection_tasks:
            task.cancel()
        await asyncio.gather(*self._connection_tasks, return_exceptions=True)
        logger.info("Market Data Feed Handler stopped.")