# quant_project/execution/broker.py
# Broker for order execution

import asyncio
import logging
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from core.event import OrderEvent, FillEvent
from core.event_loop import EventLoop
import config

logger = logging.getLogger(__name__)


class Broker:
    """
    Broker class for handling order execution with Capital.com
    """

    def __init__(self, event_queue: EventLoop):
        self.event_queue = event_queue
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # API configuration
        self.api_key = config.CAPITAL_API_KEY
        self.identifier = config.CAPITAL_IDENTIFIER
        self.password = config.CAPITAL_PASSWORD
        self.base_url = config.CAPITAL_API_URL
        self.demo_mode = config.CAPITAL_DEMO_MODE

        self.cst = None
        self.x_security_token = None
        self.session_lock = asyncio.Lock()
        self.session = requests.Session()

        logger.info(f"Broker initialized in {'DEMO' if self.demo_mode else 'LIVE'} mode")

    async def _login_sync(self):
        """Synchronous login function"""
        async with self.session_lock:
            if self.cst and self.x_security_token:
                return True

            login_url = f"{self.base_url}/session"
            headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
            payload = {"identifier": self.identifier, "password": self.password}

            try:
                response = self.session.post(login_url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    self.cst = response.headers.get("CST")
                    self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                    logger.info("[Broker] Successfully logged in to Capital.com")
                    return True
                else:
                    logger.error(
                        f"[Broker] Login failed with status {response.status_code}: {response.text}"
                    )
                    return False
            except Exception as e:
                logger.error(f"[Broker] Login exception: {e}")
                return False

    async def on_order(self, order: OrderEvent):
        """Handle order event"""
        logger.info(f"[Broker] Received order: {order.symbol} {order.direction} {order.quantity}")

        # In demo mode, simulate immediate fill
        if self.demo_mode:
            await self._simulate_fill(order)
        else:
            # Real order execution would go here
            await self._execute_order(order)

    async def _simulate_fill(self, order: OrderEvent):
        """Simulate order fill for demo mode"""
        # Simulate a small delay
        await asyncio.sleep(0.1)

        # Create fill event with simulated price
        fill_price = 100.0  # In real implementation, get current market price
        commission = order.quantity * 0.001  # 0.1% commission

        fill = FillEvent(
            symbol=order.symbol,
            timestamp=datetime.now(),
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
        )

        await self.event_queue.put_event(fill)
        logger.info(
            f"[Broker] Order filled (simulated): {fill.symbol} {fill.direction} {fill.quantity} @ {fill.fill_price}"
        )

    async def _execute_order(self, order: OrderEvent):
        """Execute real order (placeholder)"""
        # This would contain real order execution logic
        logger.warning("[Broker] Real order execution not implemented yet")
        # For now, simulate fill
        await self._simulate_fill(order)
