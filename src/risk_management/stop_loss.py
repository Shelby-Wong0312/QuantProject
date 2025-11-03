"""
Stop Loss Management Module
Fixed Stop Loss, Trailing Stop Loss, ATR Stop Loss
"""

from typing import Optional, Dict
from enum import Enum


class StopType(Enum):
    """Stop loss types"""

    FIXED = "fixed"
    TRAILING = "trailing"
    ATR = "atr"
    PERCENTAGE = "percentage"


class StopLoss:
    """Stop loss management for risk control"""

    def __init__(self):
        self.stops: Dict[str, Dict] = {}

    def set_fixed_stop(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        position_type: str = "long",
    ) -> None:
        """
        Set fixed stop loss

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_price: Stop loss price
            position_type: "long" or "short"
        """
        self.stops[symbol] = {
            "type": StopType.FIXED,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "position_type": position_type,
            "high_water_mark": entry_price if position_type == "long" else entry_price,
        }

    def set_percentage_stop(
        self,
        symbol: str,
        entry_price: float,
        stop_percentage: float,
        position_type: str = "long",
    ) -> None:
        """
        Set percentage-based stop loss

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_percentage: Stop loss percentage (e.g., 0.05 for 5%)
            position_type: "long" or "short"
        """
        if position_type == "long":
            stop_price = entry_price * (1 - stop_percentage)
        else:
            stop_price = entry_price * (1 + stop_percentage)

        self.stops[symbol] = {
            "type": StopType.PERCENTAGE,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "stop_percentage": stop_percentage,
            "position_type": position_type,
            "high_water_mark": entry_price,
        }

    def set_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        trail_amount: float,
        position_type: str = "long",
    ) -> None:
        """
        Set trailing stop loss

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            trail_amount: Trailing amount (absolute value)
            position_type: "long" or "short"
        """
        if position_type == "long":
            stop_price = entry_price - trail_amount
        else:
            stop_price = entry_price + trail_amount

        self.stops[symbol] = {
            "type": StopType.TRAILING,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "trail_amount": trail_amount,
            "position_type": position_type,
            "high_water_mark": entry_price,
        }

    def set_atr_stop(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        position_type: str = "long",
    ) -> None:
        """
        Set ATR-based stop loss

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            atr: Average True Range value
            atr_multiplier: ATR multiplier (default 2.0)
            position_type: "long" or "short"
        """
        stop_distance = atr * atr_multiplier

        if position_type == "long":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        self.stops[symbol] = {
            "type": StopType.ATR,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "atr": atr,
            "atr_multiplier": atr_multiplier,
            "position_type": position_type,
            "high_water_mark": entry_price,
        }

    def update_trailing_stop(
        self, symbol: str, current_price: float, current_high: Optional[float] = None
    ) -> bool:
        """
        Update trailing stop loss

        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_high: Current high (optional, uses current_price if not provided)

        Returns:
            True if stop was triggered, False otherwise
        """
        if symbol not in self.stops:
            return False

        stop_info = self.stops[symbol]

        if stop_info["type"] != StopType.TRAILING:
            return self.check_stop_triggered(symbol, current_price)

        position_type = stop_info["position_type"]

        if current_high is None:
            current_high = current_price

        # Update high water mark
        if position_type == "long":
            if current_high > stop_info["high_water_mark"]:
                stop_info["high_water_mark"] = current_high
                stop_info["stop_price"] = current_high - stop_info["trail_amount"]
        else:
            if current_high < stop_info["high_water_mark"]:
                stop_info["high_water_mark"] = current_high
                stop_info["stop_price"] = current_high + stop_info["trail_amount"]

        return self.check_stop_triggered(symbol, current_price)

    def check_stop_triggered(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss is triggered

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            True if stop is triggered, False otherwise
        """
        if symbol not in self.stops:
            return False

        stop_info = self.stops[symbol]
        position_type = stop_info["position_type"]
        stop_price = stop_info["stop_price"]

        if position_type == "long":
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def get_stop_info(self, symbol: str) -> Optional[Dict]:
        """Get stop loss information for a symbol"""
        return self.stops.get(symbol)

    def remove_stop(self, symbol: str) -> None:
        """Remove stop loss for a symbol"""
        if symbol in self.stops:
            del self.stops[symbol]

    @staticmethod
    def trailing_stop_loss(
        price: float,
        high: float,
        atr_multiplier: float = 2.0,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate trailing stop loss price

        Args:
            price: Current price
            high: Highest price since entry
            atr_multiplier: ATR multiplier for stop distance
            atr: ATR value (if None, uses 1% of price)

        Returns:
            Trailing stop price
        """
        if atr is None:
            atr = price * 0.01  # 1% of price as default

        stop_distance = atr * atr_multiplier
        return high - stop_distance

    @staticmethod
    def calculate_stop_distance(
        entry_price: float,
        stop_type: str = "percentage",
        stop_value: float = 0.02,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss distance

        Args:
            entry_price: Entry price
            stop_type: "percentage", "fixed", or "atr"
            stop_value: Stop value (percentage, fixed amount, or ATR multiplier)
            atr: ATR value for ATR-based stops

        Returns:
            Stop distance
        """
        if stop_type == "percentage":
            return entry_price * stop_value
        elif stop_type == "fixed":
            return stop_value
        elif stop_type == "atr" and atr is not None:
            return atr * stop_value
        else:
            return entry_price * 0.02  # Default 2% stop
