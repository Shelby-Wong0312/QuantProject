# quant_project/data/live_feed.py
# FINAL VERSION - Using 'requests' library

import asyncio
import logging
import requests # 使用 requests 函式庫
import pandas as pd
from collections import deque
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from core.event import MarketEvent
from core.event_loop import EventLoop
import config

logger = logging.getLogger(__name__)

class LiveDataFeed:
    def __init__(self, symbols: list[str], event_queue: EventLoop):
        self.symbols = symbols
        self.event_queue = event_queue
        self._running = False
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=5) # 用於在非同步環境中執行同步的requests

        # API 配置
        self.api_key = config.CAPITAL_API_KEY
        self.identifier = config.CAPITAL_IDENTIFIER
        self.password = config.CAPITAL_API_PASSWORD
        self.base_url = config.CAPITAL_BASE_URL
        self.cst = None
        self.x_security_token = None
        
        self.price_history = {symbol: deque(maxlen=200) for symbol in self.symbols}
        self.session = requests.Session() # 建立一個可複用的 session

    def _login_sync(self):
        """同步的登入函式"""
        login_url = f"{self.base_url}/session"
        headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"identifier": self.identifier, "password": self.password}
        
        try:
            response = self.session.post(login_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                logger.info("✅ [DataFeed] 成功登錄 Capital.com")
                return True
            else:
                logger.error(f"❌ [DataFeed] 登錄失敗: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ [DataFeed] 登錄時發生網路錯誤: {e}")
            return False

    def _get_market_data_sync(self, symbol: str):
        """同步的獲取市場數據函式"""
        if not self.cst: return
        
        url = f"{self.base_url}/markets/{symbol}"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}
        
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                market_data = response.json()
                snapshot = market_data.get('snapshot')
                if snapshot and snapshot.get('offer') and snapshot.get('bid'):
                    price = (snapshot['offer'] + snapshot['bid']) / 2
                    return {'symbol': symbol, 'price': price}
        except requests.exceptions.RequestException as e:
            logger.error(f"請求 {symbol} 市場數據時出錯: {e}")
        return None

    async def run(self):
        logger.info("實時數據源服務已啟動...")
        
        # 在執行器中運行同步的登入函式
        login_successful = await self.loop.run_in_executor(self.executor, self._login_sync)
        
        if not login_successful:
            logger.error("登入失敗，數據源服務無法啟動。")
            return

        self._running = True
        while self._running:
            try:
                # 建立並行任務
                tasks = [self.loop.run_in_executor(self.executor, self._get_market_data_sync, symbol) for symbol in self.symbols]
                results = await asyncio.gather(*tasks)

                for result in results:
                    if result:
                        symbol = result['symbol']
                        price = result['price']
                        timestamp = datetime.now(timezone.utc)
                        
                        self.price_history[symbol].append({'Date': timestamp, 'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 0})
                        
                        if len(self.price_history[symbol]) >= 50:
                            df = pd.DataFrame(list(self.price_history[symbol])).set_index('Date')
                            market_event = MarketEvent(symbol=symbol, timestamp=timestamp, ohlcv_data=df)
                            await self.event_queue.put_event(market_event)
                            logger.debug(f"已發送 {symbol} 的市場數據事件")
                
                await asyncio.sleep(2) # 每2秒輪詢一次
            except asyncio.CancelledError:
                self._running = False
        
        logger.info("實時數據源服務已停止。")

    def stop(self):
        self._running = False