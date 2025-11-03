"""
Order Manager - Handles order execution and management
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderManager:
    """Manages order execution and tracking"""

    def __init__(self):
        self.orders = []
        self.order_id_counter = 0
        self.pending_orders = []
        self.filled_orders = []

    async def submit_order(self, order: Dict) -> Dict:
        """Submit new order"""
        # Add order ID
        self.order_id_counter += 1
        order["id"] = self.order_id_counter
        order["status"] = OrderStatus.PENDING.value
        order["submitted_at"] = datetime.now()

        # Add to pending orders
        self.pending_orders.append(order)
        self.orders.append(order)

        logger.info(
            f"Order {order['id']} submitted: {order['symbol']} {order['side']} {order['quantity']}"
        )

        return order

    async def execute_order(self, order: Dict) -> Dict:
        """Execute order (simulated)"""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)

            # For demo, assume all orders fill
            order["status"] = OrderStatus.FILLED.value
            order["filled_at"] = datetime.now()
            order["price"] = self._get_execution_price(order)

            # Move from pending to filled
            if order in self.pending_orders:
                self.pending_orders.remove(order)
            self.filled_orders.append(order)

            logger.info(f"Order {order['id']} filled at ${order['price']:.2f}")

            return {"status": "filled", "order": order}

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order["status"] = OrderStatus.REJECTED.value
            return {"status": "rejected", "order": order}

    def _get_execution_price(self, order: Dict) -> float:
        """Get simulated execution price"""
        # In real system, this would connect to broker
        # For demo, return a simulated price
        import random

        base_price = 100.0  # Simulated
        slippage = random.uniform(-0.01, 0.01)  # 1% slippage
        return base_price * (1 + slippage)

    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders"""
        return [
            o for o in self.pending_orders if o["status"] == OrderStatus.PENDING.value
        ]

    async def cancel_order(self, order_id: int) -> bool:
        """Cancel specific order"""
        for order in self.pending_orders:
            if order["id"] == order_id:
                order["status"] = OrderStatus.CANCELLED.value
                order["cancelled_at"] = datetime.now()
                self.pending_orders.remove(order)
                logger.info(f"Order {order_id} cancelled")
                return True
        return False

    async def cancel_all_orders(self):
        """Cancel all pending orders"""
        for order in self.pending_orders:
            order["status"] = OrderStatus.CANCELLED.value
            order["cancelled_at"] = datetime.now()

        cancelled_count = len(self.pending_orders)
        self.pending_orders.clear()

        logger.info(f"Cancelled {cancelled_count} pending orders")

    def get_order_stats(self) -> Dict:
        """Get order statistics"""
        return {
            "total_orders": len(self.orders),
            "pending": len(self.pending_orders),
            "filled": len(self.filled_orders),
            "cancelled": len(
                [o for o in self.orders if o["status"] == OrderStatus.CANCELLED.value]
            ),
            "rejected": len(
                [o for o in self.orders if o["status"] == OrderStatus.REJECTED.value]
            ),
        }
