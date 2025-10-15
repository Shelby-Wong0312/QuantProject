"""
Risk Monitor - Real-time risk monitoring and alerts
"""

import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class RiskMonitor:
    """Real-time risk monitoring system"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_alerts = []
        self.max_positions = config.get("max_positions", 10)
        self.max_single_position = config.get("max_single_position", 0.2)
        self.daily_loss_limit = config.get("daily_loss_limit", 0.05)
        self.stop_loss = config.get("stop_loss", 0.02)

    def can_take_position(self, positions: Dict, capital: float) -> bool:
        """Check if new position can be taken"""
        # Check max positions
        if len(positions) >= self.max_positions:
            logger.warning("Max positions reached")
            return False

        # Check capital allocation
        total_allocated = sum(p.get("value", 0) for p in positions.values())
        if total_allocated / capital > 0.95:  # 95% capital limit
            logger.warning("Capital allocation limit reached")
            return False

        return True

    async def check_stop_losses(self, positions: Dict):
        """Check and trigger stop losses"""
        for symbol, position in positions.items():
            if "current_price" in position and "entry_price" in position:
                loss_pct = (position["current_price"] - position["entry_price"]) / position[
                    "entry_price"
                ]

                if position["quantity"] > 0:  # Long position
                    if loss_pct < -self.stop_loss:
                        await self._trigger_stop_loss(symbol, position)
                else:  # Short position
                    if loss_pct > self.stop_loss:
                        await self._trigger_stop_loss(symbol, position)

    async def _trigger_stop_loss(self, symbol: str, position: Dict):
        """Trigger stop loss for position"""
        alert = {
            "timestamp": datetime.now(),
            "type": "stop_loss",
            "symbol": symbol,
            "message": f"Stop loss triggered for {symbol}",
            "position": position,
        }
        self.risk_alerts.append(alert)
        logger.critical(f"STOP LOSS triggered for {symbol}")

    def check_position_limits(self, positions: Dict, capital: float) -> List[str]:
        """Check position size limits"""
        oversized = []

        for symbol, position in positions.items():
            if abs(position.get("value", 0)) / capital > self.max_single_position:
                oversized.append(symbol)
                logger.warning(f"Position limit exceeded for {symbol}")

        return oversized

    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0

        import numpy as np

        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return sorted_returns[index] if index < len(sorted_returns) else sorted_returns[0]

    def get_risk_metrics(self, positions: Dict, capital: float) -> Dict:
        """Get current risk metrics"""
        total_exposure = sum(abs(p.get("value", 0)) for p in positions.values())

        return {
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / capital if capital > 0 else 0,
            "position_count": len(positions),
            "max_position_pct": (
                max(abs(p.get("value", 0)) / capital for p in positions.values())
                if positions
                else 0
            ),
            "alerts_count": len(self.risk_alerts),
        }

    def clear_alerts(self):
        """Clear risk alerts"""
        self.risk_alerts.clear()
