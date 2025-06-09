# data_feeds/feed_handler.py
import asyncio
import websockets
import json
import logging
import pandas as pd
from core.event import MarketDataEvent # Assuming your event definitions

logger = logging.getLogger(__name__)

class AsyncMarketDataFeedHandler:
    def __init__(self, symbols: list, api_key: str, event_queue: asyncio.Queue, provider_url: str):
        self.symbols = symbols
        self.api_key = api_key
        self.event_queue = event_queue
        self.provider_url = provider_url
        self._connection_tasks = []
        self._running = False
        self.MAX_SYMBOLS_PER_CONNECTION = 500

    async def _handle_connection(self, symbols_for_this_connection):
        async for websocket in websockets.connect(self.provider_url, ping_interval=20, ping_timeout=20):
            try:
                # 1. Authenticate
                auth_payload = {"action": "auth", "params": self.api_key}
                await websocket.send(json.dumps(auth_payload))
                auth_response_str = await websocket.recv()
                
                # --- 強化: 認證回應檢查 ---
                try:
                    auth_data = json.loads(auth_response_str)
                    if isinstance(auth_data, list) and auth_data[0].get("status") != "auth_success":
                        logger.critical(f"Authentication failed: {auth_data[0].get('message')}. Stopping this connection.")
                        break # 認證失敗，不再重連
                except (json.JSONDecodeError, IndexError, AttributeError) as e:
                    logger.error(f"Failed to parse auth response: {auth_response_str}. Error: {e}")
                    continue # 嘗試重連

                logger.info(f"Auth successful for {symbols_for_this_connection[:5]}...")

                # 2. Subscribe to symbols
                subscribe_payload = {"action": "subscribe", "params": ",".join(symbols_for_this_connection)}
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Subscribed to {len(symbols_for_this_connection)} symbols on one connection.")

                # 3. Receive messages
                while self._running:
                    message_str = await websocket.recv()
                    
                    # --- 強化: JSON 解析安全 ---
                    try:
                        messages = json.loads(message_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Malformed JSON received: {message_str}")
                        continue

                    for msg_data in messages:
                        # --- 強化: 數據格式驗證 ---
                        if not isinstance(msg_data, dict):
                            logger.warning(f"Received message is not a dictionary: {msg_data}")
                            continue

                        event_type = msg_data.get('ev')
                        if not event_type:
                            logger.warning(f"Message without 'ev' type received: {msg_data}")
                            continue
                        
                        if event_type == 'T': # Trade event
                            symbol = msg_data.get('sym')
                            timestamp_ms = msg_data.get('t')
                            price = msg_data.get('p')
                            size = msg_data.get('s')

                            if not all([symbol, timestamp_ms, price, size]):
                                logger.warning(f"Trade message with missing fields: {msg_data}")
                                continue
                            
                            try:
                                market_event = MarketDataEvent(
                                    symbol=symbol,
                                    timestamp=pd.to_datetime(timestamp_ms, unit='ms'),
                                    # ... populate other fields like price, volume
                                )
                                await self.event_queue.put(market_event)
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error creating MarketDataEvent for trade: {e}, data: {msg_data}")

            except websockets.exceptions.ConnectionClosed as e:
                if self._running:
                    logger.error(f"Connection closed for symbols {symbols_for_this_connection[:5]}...: {e}. Reconnecting...")
                    await asyncio.sleep(5)
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Unexpected error in connection handler: {e}", exc_info=True)
                    await asyncio.sleep(15)

    def start(self):
        if not self._running:
            self._running = True
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