"""
Dynamic Stop Loss System
動態止損系統
Cloud Quant - Task Q-601
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """持倉類型"""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class StopLossInfo:
    """止損信息"""

    stop_price: float
    entry_price: float
    position_type: PositionType
    trailing_activated: bool
    highest_price: float
    lowest_price: float
    entry_time: datetime
    last_update: datetime


class DynamicStopLoss:
    """
    動態止損管理器
    支持多種止損策略：ATR、百分比、追蹤止損、時間止損
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        trailing_percent: float = 0.05,
        time_stop_hours: int = 24,
        profit_lock_levels: List[float] = None,
    ):
        """
        初始化動態止損管理器

        Args:
            atr_multiplier: ATR倍數
            trailing_percent: 追蹤止損百分比
            time_stop_hours: 時間止損小時數
            profit_lock_levels: 利潤鎖定級別
        """
        self.atr_multiplier = atr_multiplier
        self.trailing_percent = trailing_percent
        self.time_stop_hours = time_stop_hours
        self.profit_lock_levels = profit_lock_levels or [0.25, 0.5, 0.75]

        # 持倉止損信息
        self.position_stops: Dict[str, StopLossInfo] = {}

        # 統計信息
        self.stats = {
            "total_stops_set": 0,
            "stops_triggered": 0,
            "trailing_updates": 0,
            "profit_locks": 0,
        }

        logger.info(
            f"Dynamic Stop Loss initialized - ATR: {atr_multiplier}x, Trailing: {trailing_percent:.1%}"
        )

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        計算 Average True Range

        Args:
            df: OHLC數據
            period: ATR週期

        Returns:
            ATR序列
        """
        if len(df) < period:
            logger.warning(f"Insufficient data for ATR calculation (need {period} rows)")
            return pd.Series([df["high"].iloc[-1] - df["low"].iloc[-1]] * len(df))

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range 計算
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # 填充初始NaN值
        atr = atr.fillna(method="bfill")

        return atr

    def set_initial_stop(
        self, symbol: str, entry_price: float, current_atr: float, position_type: str = "LONG"
    ) -> float:
        """
        設置初始止損價

        Args:
            symbol: 股票代碼
            entry_price: 進場價格
            current_atr: 當前ATR值
            position_type: 持倉類型

        Returns:
            止損價格
        """
        pos_type = PositionType[position_type.upper()]

        # 計算止損價
        if pos_type == PositionType.LONG:
            stop_price = entry_price - (current_atr * self.atr_multiplier)
        else:  # SHORT
            stop_price = entry_price + (current_atr * self.atr_multiplier)

        # 創建止損信息
        stop_info = StopLossInfo(
            stop_price=stop_price,
            entry_price=entry_price,
            position_type=pos_type,
            trailing_activated=False,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_time=datetime.now(),
            last_update=datetime.now(),
        )

        self.position_stops[symbol] = stop_info
        self.stats["total_stops_set"] += 1

        logger.info(f"{symbol}: Initial stop set at {stop_price:.2f} (Entry: {entry_price:.2f})")

        return stop_price

    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """
        更新追蹤止損

        Args:
            symbol: 股票代碼
            current_price: 當前價格

        Returns:
            更新後的止損價格
        """
        if symbol not in self.position_stops:
            return None

        stop_info = self.position_stops[symbol]
        old_stop = stop_info.stop_price

        if stop_info.position_type == PositionType.LONG:
            # 更新最高價
            if current_price > stop_info.highest_price:
                stop_info.highest_price = current_price

                # 計算新的追蹤止損價
                new_stop = current_price * (1 - self.trailing_percent)

                # 只允許止損價上移
                if new_stop > stop_info.stop_price:
                    stop_info.stop_price = new_stop
                    stop_info.trailing_activated = True
                    stop_info.last_update = datetime.now()
                    self.stats["trailing_updates"] += 1

                    logger.debug(
                        f"{symbol}: Trailing stop updated {old_stop:.2f} -> {new_stop:.2f}"
                    )

        else:  # SHORT position
            # 更新最低價
            if current_price < stop_info.lowest_price:
                stop_info.lowest_price = current_price

                # 計算新的追蹤止損價
                new_stop = current_price * (1 + self.trailing_percent)

                # 只允許止損價下移
                if new_stop < stop_info.stop_price:
                    stop_info.stop_price = new_stop
                    stop_info.trailing_activated = True
                    stop_info.last_update = datetime.now()
                    self.stats["trailing_updates"] += 1

                    logger.debug(
                        f"{symbol}: Trailing stop updated {old_stop:.2f} -> {new_stop:.2f}"
                    )

        return stop_info.stop_price

    def check_stop_triggered(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        檢查是否觸發止損

        Args:
            symbol: 股票代碼
            current_price: 當前價格

        Returns:
            (是否觸發, 觸發原因)
        """
        if symbol not in self.position_stops:
            return False, ""

        stop_info = self.position_stops[symbol]

        # 價格止損檢查
        if stop_info.position_type == PositionType.LONG:
            if current_price <= stop_info.stop_price:
                self.stats["stops_triggered"] += 1
                return True, f"Price stop triggered at {current_price:.2f}"
        else:  # SHORT
            if current_price >= stop_info.stop_price:
                self.stats["stops_triggered"] += 1
                return True, f"Price stop triggered at {current_price:.2f}"

        # 時間止損檢查
        if self.check_time_stop(symbol):
            self.stats["stops_triggered"] += 1
            return True, f"Time stop triggered after {self.time_stop_hours} hours"

        return False, ""

    def check_time_stop(self, symbol: str) -> bool:
        """
        檢查時間止損

        Args:
            symbol: 股票代碼

        Returns:
            是否觸發時間止損
        """
        if symbol not in self.position_stops:
            return False

        stop_info = self.position_stops[symbol]
        holding_duration = (datetime.now() - stop_info.entry_time).total_seconds() / 3600

        return holding_duration >= self.time_stop_hours

    def calculate_profit_target(
        self, entry_price: float, stop_price: float, risk_reward_ratio: float = 2.0
    ) -> float:
        """
        計算獲利目標價

        Args:
            entry_price: 進場價格
            stop_price: 止損價格
            risk_reward_ratio: 風險回報比

        Returns:
            獲利目標價
        """
        risk_amount = abs(entry_price - stop_price)
        profit_target = entry_price + (risk_amount * risk_reward_ratio)

        return profit_target

    def update_profit_lock(self, symbol: str, current_price: float) -> Optional[float]:
        """
        更新利潤鎖定止損

        Args:
            symbol: 股票代碼
            current_price: 當前價格

        Returns:
            更新後的止損價格
        """
        if symbol not in self.position_stops:
            return None

        stop_info = self.position_stops[symbol]

        # 計算當前利潤
        if stop_info.position_type == PositionType.LONG:
            profit_pct = (current_price - stop_info.entry_price) / stop_info.entry_price

            # 根據利潤級別調整止損
            for level in sorted(self.profit_lock_levels, reverse=True):
                if profit_pct >= level:
                    # 鎖定部分利潤
                    lock_price = stop_info.entry_price * (1 + level * 0.5)

                    if lock_price > stop_info.stop_price:
                        stop_info.stop_price = lock_price
                        self.stats["profit_locks"] += 1
                        logger.info(
                            f"{symbol}: Profit lock at {level:.0%} - Stop: {lock_price:.2f}"
                        )
                    break

        return stop_info.stop_price

    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """
        獲取持倉止損信息

        Args:
            symbol: 股票代碼

        Returns:
            止損信息字典
        """
        if symbol not in self.position_stops:
            return None

        stop_info = self.position_stops[symbol]

        return {
            "symbol": symbol,
            "stop_price": stop_info.stop_price,
            "entry_price": stop_info.entry_price,
            "position_type": stop_info.position_type.value,
            "trailing_activated": stop_info.trailing_activated,
            "highest_price": stop_info.highest_price,
            "lowest_price": stop_info.lowest_price,
            "holding_hours": (datetime.now() - stop_info.entry_time).total_seconds() / 3600,
            "last_update": stop_info.last_update.isoformat(),
        }

    def remove_position(self, symbol: str):
        """
        移除持倉止損信息

        Args:
            symbol: 股票代碼
        """
        if symbol in self.position_stops:
            del self.position_stops[symbol]
            logger.info(f"{symbol}: Position stop removed")

    def get_statistics(self) -> Dict:
        """
        獲取統計信息

        Returns:
            統計信息字典
        """
        return {
            "active_stops": len(self.position_stops),
            "total_stops_set": self.stats["total_stops_set"],
            "stops_triggered": self.stats["stops_triggered"],
            "trailing_updates": self.stats["trailing_updates"],
            "profit_locks": self.stats["profit_locks"],
            "trigger_rate": self.stats["stops_triggered"] / max(1, self.stats["total_stops_set"]),
        }


class ProfitProtection:
    """
    利潤保護機制
    動態調整止損以保護利潤
    """

    def __init__(self, lock_levels: List[Tuple[float, float]] = None):
        """
        初始化利潤保護

        Args:
            lock_levels: [(利潤百分比, 保護百分比)]
        """
        self.lock_levels = lock_levels or [
            (0.1, 0.5),  # 10%利潤時保護50%
            (0.2, 0.75),  # 20%利潤時保護75%
            (0.3, 0.9),  # 30%利潤時保護90%
        ]
        self.protection_history = {}

    def calculate_protection_stop(
        self, entry_price: float, current_price: float, position_type: str = "LONG"
    ) -> float:
        """
        計算利潤保護止損價

        Args:
            entry_price: 進場價格
            current_price: 當前價格
            position_type: 持倉類型

        Returns:
            保護止損價
        """
        if position_type == "LONG":
            profit = current_price - entry_price
            profit_pct = profit / entry_price

            # 找到適用的保護級別
            protection_price = entry_price  # 默認保本

            for profit_threshold, lock_pct in self.lock_levels:
                if profit_pct >= profit_threshold:
                    # 保護指定百分比的利潤
                    protection_price = entry_price + (profit * lock_pct)

            return protection_price

        else:  # SHORT
            profit = entry_price - current_price
            profit_pct = profit / entry_price

            protection_price = entry_price  # 默認保本

            for profit_threshold, lock_pct in self.lock_levels:
                if profit_pct >= profit_threshold:
                    protection_price = entry_price - (profit * lock_pct)

            return protection_price

    def should_update_stop(
        self, current_stop: float, new_stop: float, position_type: str = "LONG"
    ) -> bool:
        """
        判斷是否應該更新止損

        Args:
            current_stop: 當前止損價
            new_stop: 新止損價
            position_type: 持倉類型

        Returns:
            是否更新
        """
        if position_type == "LONG":
            # LONG position - 止損只能上移
            return new_stop > current_stop
        else:
            # SHORT position - 止損只能下移
            return new_stop < current_stop


if __name__ == "__main__":
    # 測試動態止損系統
    print("Testing Dynamic Stop Loss System...")
    print("=" * 50)

    # 創建測試數據
    test_data = pd.DataFrame(
        {
            "high": [100, 102, 101, 103, 104, 102, 100, 98],
            "low": [98, 99, 100, 101, 102, 99, 97, 95],
            "close": [99, 101, 100.5, 102, 103, 100, 98, 96],
        }
    )

    # 初始化止損管理器
    sl_manager = DynamicStopLoss(atr_multiplier=2.0)

    # 計算 ATR
    atr = sl_manager.calculate_atr(test_data)
    print(f"ATR values: {atr.values[-3:]}")

    # 設置初始止損
    entry_price = 100
    stop_price = sl_manager.set_initial_stop("TEST", entry_price, atr.iloc[-1])

    print(f"\nEntry Price: {entry_price}")
    print(f"Initial Stop: {stop_price:.2f}")

    # 模擬價格變動
    print("\nSimulating price movements:")
    print("-" * 40)

    price_sequence = [101, 102, 103, 104, 103, 102, 101, 99, 98]

    for price in price_sequence:
        # 更新追蹤止損
        new_stop = sl_manager.update_trailing_stop("TEST", price)

        # 檢查是否觸發
        triggered, reason = sl_manager.check_stop_triggered("TEST", price)

        print(f"Price: {price:>6.2f} | Stop: {new_stop:>6.2f} | Triggered: {triggered}")

        if triggered:
            print(f"\n*** STOP LOSS TRIGGERED! ***")
            print(f"Reason: {reason}")
            break

    # 顯示統計
    print("\n" + "=" * 50)
    print("Statistics:")
    stats = sl_manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 測試利潤保護
    print("\n" + "=" * 50)
    print("Testing Profit Protection:")

    profit_protector = ProfitProtection()

    test_prices = [100, 110, 120, 130, 125, 120]
    entry = 100

    for price in test_prices:
        protection_stop = profit_protector.calculate_protection_stop(entry, price)
        profit_pct = (price - entry) / entry
        print(f"Price: {price} | Profit: {profit_pct:.1%} | Protection Stop: {protection_stop:.2f}")

    print("\nDynamic Stop Loss System Test Complete!")
