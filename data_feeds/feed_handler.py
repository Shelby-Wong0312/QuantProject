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
    # __init__ 方法現在接收 Alpaca 的金鑰
    def __init__(self, symbols: list, api_key_id: str, secret_key: str, event_queue: asyncio.Queue, provider_url: str):
        self.symbols = symbols
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.event_queue = event_queue
        self.provider_url = provider_url
        self._connection_task = None # Alpaca 建議單一連接
        self._running = False

    async def _handle_connection(self):
        async for websocket in websockets.connect(self.provider_url):
            try:
                # 1. 等待連接成功的 "success" 消息
                conn_msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                logger.info(f"Connection status: {conn_msg}")

                # 2. 發送認證請求
                auth_payload = {
                    "action": "auth",
                    "key": self.api_key_id,
                    "secret": self.secret_key
                }
                await websocket.send(json.dumps(auth_payload))
                
                # 3. 等待認證結果
                auth_resp = await asyncio.wait_for(websocket.recv(), timeout=10)
                logger.info(f"Auth response: {auth_resp}")
                # TODO: 檢查認證是否成功

                # 4. 發送訂閱請求
                subscribe_payload = {
                    "action": "subscribe",
                    "trades": self.symbols,
                    # "quotes": self.symbols # 也可以訂閱報價
                }
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Subscription sent for trades: {self.symbols}")

                # 5. 接收數據
                while self._running:
                    message_str = await websocket.recv()
                    messages = json.loads(message_str)
                    
                    for msg_data in messages:
                        # Alpaca 的交易數據類型為 't'
                        if msg_data.get('T') == 't':
                            market_event = MarketDataEvent(
                                symbol=msg_data.get('S'),
                                timestamp=pd.to_datetime(msg_data.get('t')) # Alpaca 的時間戳是 RFC-3339 格式
                            )
                            await self.event_queue.put(market_event)

            except Exception as e:
                logger.error(f"Error in Alpaca connection handler: {e}", exc_info=True)
                if self._running:
                    await asyncio.sleep(5) # 等待後重連
                continue

    def start(self):
        if not self._running:
            self._running = True
            # Alpaca 的 IEX 源通常使用單一連接即可
            self._connection_task = asyncio.create_task(self._handle_connection())
            logger.info("Alpaca Market Data Feed Handler started.")

    async def stop(self):
        self._running = False
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        logger.info("Alpaca Market Data Feed Handler stopped.")