# data_feeds/feed_handler.py

import asyncio
import logging
from typing import List
from datetime import datetime, timezone

# 假設使用官方的 polygon-io 客戶端庫
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market

from core.event_types import MarketDataEvent
from core import config

logger = logging.getLogger(__name__)

class FeedHandler:
    """
    負責連接 Polygon.io WebSocket，接收延遲市場數據，並轉換為 MarketDataEvent。
    """
    def __init__(self,
                 market_data_queue: asyncio.Queue,
                 symbols: List[str],
                 api_key: str):
        self.market_data_queue = market_data_queue
        self.symbols = symbols
        self.api_key = api_key
        self.ws_client = WebSocketClient(
            api_key=self.api_key,
            market=Market.Stocks,
            feed="socket.polygon.io", # 指定 feed
            subscriptions=self._prepare_subscriptions()
        )
        self._running = False

    def _prepare_subscriptions(self) -> List[str]:
        """
        準備 Polygon.io 的訂閱列表。
        Stocks Starter 方案主要關注分鐘聚合數據。
        """
        # 訂閱格式為 "AM.TICKER" (Aggregates per Minute for TICKER)
        return [f"AM.{symbol}" for symbol in self.symbols]

    async def _message_handler(self, messages: List[WebSocketMessage]):
        """
        異步處理從 WebSocket 收到的消息。
        """
        for msg in messages:
            # 我們只關心分鐘聚合數據 (event_type == "AM")
            if msg.event_type == "AM":
                try:
                    # Polygon.io 的時間戳是 Unix 毫秒，需要轉換
                    event_timestamp = datetime.fromtimestamp(msg.end_timestamp / 1000.0, tz=timezone.utc)
                    
                    market_event = MarketDataEvent(
                        symbol=msg.symbol,
                        timestamp=event_timestamp,
                        event_type="AGG_MINUTE",
                        data={
                            "open": msg.open,
                            "high": msg.high,
                            "low": msg.low,
                            "close": msg.close,
                            "volume": msg.volume,
                            "vwap": msg.vwap,
                            "start_ts": msg.start_timestamp,
                            "end_ts": msg.end_timestamp
                        }
                    )
                    await self.market_data_queue.put(market_event)
                    logger.debug(f"Published MarketDataEvent for {msg.symbol}")
                except Exception as e:
                    logger.error(f"Error processing message for {msg.symbol}: {msg} - {e}", exc_info=True)
            elif msg.event_type == "status":
                logger.info(f"Polygon status message: {msg.message} (Status: {msg.status})")

    async def run(self, shutdown_event: asyncio.Event):
        self._running = True
        logger.info("FeedHandler (Polygon.io) starting...")
        loop = asyncio.get_running_loop()

        while self._running and not shutdown_event.is_set():
            logger.info("Attempting to connect to Polygon.io WebSocket...")
            try:
                # ws_client.run() 是一個阻塞操作，我們將它放入背景執行緒
                # handle_msg 參數指定了收到消息時要調用的回調函式
                await loop.run_in_executor(None, self.ws_client.run, self._message_handler)

                # 如果 run() 正常退出，通常意味著連接已斷開
                if not shutdown_event.is_set():
                    logger.warning("Polygon.io stream disconnected. Reconnecting...")
                    await asyncio.sleep(5) # 重連前等待5秒
            
            except Exception as e:
                logger.error(f"FeedHandler (Polygon.io) error: {e}", exc_info=True)
                if not shutdown_event.is_set():
                    logger.info("Attempting to reconnect in 10 seconds...")
                    await asyncio.sleep(10)
        
        logger.info("FeedHandler shutting down...")

    def stop(self):
        """
        停止 FeedHandler。
        """
        logger.info("Stopping FeedHandler (Polygon.io)...")
        self._running = False
        if self.ws_client:
            self.ws_client.close()
            logger.info("Polygon.io WebSocket client closed.")