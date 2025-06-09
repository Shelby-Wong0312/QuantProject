# data_feeds/feed_handler.py
import asyncio
import websockets
import json
import logging
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketDataEvent:
    symbol: str
    timestamp: datetime

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
        # 啟用 websockets 函式庫的詳細偵錯日誌
        logging.getLogger("websockets").setLevel(logging.DEBUG)

        async for websocket in websockets.connect(self.provider_url, ping_interval=20, ping_timeout=20):
            try:
                logger.info(f"Attempting to connect for symbols: {symbols_for_this_connection[:5]}...")
                
                # 1. Authenticate
                auth_payload = {"action": "auth", "params": self.api_key}
                logger.debug(f"--> SENDING AUTH: {json.dumps(auth_payload)}")
                await websocket.send(json.dumps(auth_payload))
                
                auth_response_str = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                logger.debug(f"<-- RECEIVED AUTH RESP: {auth_response_str}")

                try:
                    auth_data = json.loads(auth_response_str)
                    if isinstance(auth_data, list) and auth_data[0].get("status") == "auth_success":
                        logger.info("Authentication successful.")
                    else:
                        logger.critical(f"Authentication FAILED. Response: {auth_response_str}. Stopping this connection.")
                        break 
                except (json.JSONDecodeError, IndexError, AttributeError) as e:
                    logger.error(f"Failed to parse auth response: {auth_response_str}. Error: {e}")
                    continue

                # 2. Subscribe to symbols
                subscribe_payload = {"action": "subscribe", "params": ",".join(symbols_for_this_connection)}
                logger.debug(f"--> SENDING SUBSCRIBE: {json.dumps(subscribe_payload)}")
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Subscription sent for {len(symbols_for_this_connection)} symbols.")

                # 3. Receive messages
                while self._running:
                    try:
                        message_str = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                        logger.debug(f"<-- RECEIVED DATA: {message_str}")
                    except asyncio.TimeoutError:
                        logger.warning("No message received in 60 seconds. Connection might be stale. Pinging...")
                        # 主動發送 ping 來確認連線狀態
                        try:
                            await websocket.ping()
                        except websockets.exceptions.ConnectionClosed:
                            logger.error("Connection is closed, will attempt to reconnect.")
                            break # 跳出內層循環，外層的 async for 會處理重連
                        continue

                    # (後續的 JSON 解析與事件處理邏輯不變)
                    messages = json.loads(message_str)
                    for msg_data in messages:
                         if msg_data.get('ev') == 'T':
                            market_event = MarketDataEvent(
                                symbol=msg_data.get('sym'),
                                timestamp=pd.to_datetime(msg_data.get('t'), unit='ms')
                            )
                            await self.event_queue.put(market_event)

            except websockets.exceptions.ConnectionClosed as e:
                if self._running:
                    logger.error(f"Connection closed: {e}. Reconnecting...")
                    await asyncio.sleep(5)
                continue
            except asyncio.TimeoutError:
                logger.error("Connection timed out during auth. Retrying...")
                await asyncio.sleep(5)
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
        # (stop 方法的其餘部分不變)
        logger.info("Stopping Market Data Feed Handler...")
        for task in self._connection_tasks:
            task.cancel()
        await asyncio.gather(*self._connection_tasks, return_exceptions=True)
        logger.info("Market Data Feed Handler stopped.")