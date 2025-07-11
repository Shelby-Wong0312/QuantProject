# quant_project/execution/broker.py
# FINAL FIX - Missing Import

import asyncio
import logging
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from core.event import OrderEvent, FillEvent
from core.event_loop import EventLoop
import config # <--- 新增這一行，解決 NameError

logger = logging.getLogger(__name__)

class Broker:
    """
    使用 requests 函式庫處理與 Capital.com 的訂單執行。
    """
    def __init__(self, event_queue: EventLoop):
        self.event_queue = event_queue
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # API 配置
        self.api_key = config.CAPITAL_API_KEY
        self.identifier = config.CAPITAL_IDENTIFIER
        self.password = config.CAPITAL_API_PASSWORD
        self.base_url = config.CAPITAL_BASE_URL
        self.cst = None
        self.x_security_token = None
        self.session_lock = asyncio.Lock()
        self.session = requests.Session()

    async def _login_sync(self):
        """同步的登入函式"""
        async with self.session_lock:
            if self.cst and self.x_security_token: return True
            
            login_url = f"{self.base_url}/session"
            headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
            payload = {"identifier": self.identifier, "password": self.password}
            
            try:
                response = self.session.post(login_url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    self.cst = response.headers.get("CST")
                    self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                    logger.info("✅ [Broker] 成功登錄 Capital.com")
                    return True
                logger.error(f"❌ [Broker] 登錄失敗: {response.text}")
                return False
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ [Broker] 登錄時發生網路錯誤: {e}")
                return False

    def _place_order_sync(self, order: OrderEvent):
        """同步的下單函式"""
        url = f"{self.base_url}/positions"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token, "Content-Type": "application/json"}
        payload = {"epic": order.symbol, "direction": order.direction.upper(), "size": order.quantity}
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200 and response.json().get("dealReference"):
                logger.info(f"✅ [Broker] 訂單成功發送: {order.direction} {order.quantity} {order.symbol}")
                return self._get_fill_price_sync(order)
            else:
                logger.error(f"❌ [Broker] 訂單失敗: {response.status} - {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ [Broker] 下單時發生網路錯誤: {e}")
        return None

    def _get_fill_price_sync(self, order: OrderEvent):
        """同步的獲取成交價函式"""
        price_url = f"{self.base_url}/markets/{order.symbol}"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}
        try:
            response = self.session.get(price_url, headers=headers, timeout=10)
            if response.status_code == 200:
                price = response.json().get('snapshot', {}).get('bid' if order.direction.upper() == 'SELL' else 'offer')
                return float(price)
        except requests.exceptions.RequestException as e:
            logger.error(f"創建成交事件時獲取價格失敗: {e}")
        return None

    async def on_order(self, order: OrderEvent):
        """接收訂單事件，並在執行器中運行同步的下單函式"""
        if not (self.cst and self.x_security_token):
             if not await self.loop.run_in_executor(self.executor, self._login_sync):
                logger.error(f"下單失敗 {order.symbol}: 無法登錄。")
                return

        fill_price = await self.loop.run_in_executor(self.executor, self._place_order_sync, order)
        
        if fill_price is not None:
            fill_event = FillEvent(
                symbol=order.symbol,
                timestamp=datetime.now(),
                direction=order.direction,
                quantity=order.quantity,
                fill_price=fill_price
            )
            await self.event_queue.put_event(fill_event)