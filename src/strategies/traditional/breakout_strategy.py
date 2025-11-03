"""
突破策略 - 基於價格突破關鍵阻力支撑位
使用通道突破、成交量確認、ATR動態止損
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from ..base_strategy import BaseStrategy
from ..strategy_interface import TradingSignal, SignalType, StrategyConfig, Position
from ...indicators.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    突破策略 - 捕捉價格突破關鍵位點

    核心邏輯：
    - 價格突破阻力位 + 成交量放大 = 買入
    - 價格跌破支撐位 + 成交量放大 = 賣出
    - ATR動態止損，避免假突破
    """

    def _initialize_parameters(self) -> None:
        """初始化策略參數"""
        params = self.config.parameters

        # 通道參數
        self.channel_period = params.get("channel_period", 20)
        self.breakout_threshold = params.get("breakout_threshold", 0.001)  # 0.1%突破確認

        # 成交量參數
        self.volume_period = params.get("volume_period", 20)
        self.volume_multiplier = params.get("volume_multiplier", 1.5)

        # ATR參數
        self.atr_period = params.get("atr_period", 14)
        self.atr_stop_multiplier = params.get("atr_stop_multiplier", 2.0)

        # 趨勢確認參數
        self.ma_fast = params.get("ma_fast", 10)
        self.ma_slow = params.get("ma_slow", 20)

        # 風險參數
        self.position_size_pct = params.get("position_size_pct", 0.12)
        self.max_risk_per_trade = params.get("max_risk_per_trade", 0.02)
        self.profit_target_atr = params.get("profit_target_atr", 3.0)

        # 過濾參數
        self.min_breakout_bars = params.get("min_breakout_bars", 3)
        self.consolidation_period = params.get("consolidation_period", 10)

        self.indicator_calc = IndicatorCalculator()

        logger.info(
            f"{self.name}: Initialized with Channel({self.channel_period}), "
            f"ATR({self.atr_period}), Volume({self.volume_period})"
        )

    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        計算突破交易信號

        Args:
            data: OHLCV數據

        Returns:
            交易信號列表
        """
        []

        if len(data) < max(self.channel_period, self.volume_period, self.ma_slow) + 10:
            return signals

        try:
            # 計算技術指標
            close_prices = data["close"]
            high_prices = data["high"]
            low_prices = data["low"]
            volumes = data["volume"]

            # 計算通道 (Donchian Channel)
            channel_high = high_prices.rolling(window=self.channel_period).max()
            channel_low = low_prices.rolling(window=self.channel_period).min()
            channel_mid = (channel_high + channel_low) / 2

            # ATR
            atr = self.indicator_calc.calculate_atr(
                high_prices, low_prices, close_prices, self.atr_period
            )

            # 成交量均線
            volume_ma = volumes.rolling(window=self.volume_period).mean()

            # 趨勢確認均線
            ma_fast = close_prices.rolling(window=self.ma_fast).mean()
            ma_slow = close_prices.rolling(window=self.ma_slow).mean()

            # 最新值
            current_price = close_prices.iloc[-1]
            current_high = high_prices.iloc[-1]
            current_low = low_prices.iloc[-1]
            current_volume = volumes.iloc[-1]

            current_channel_high = channel_high.iloc[-1]
            current_channel_low = channel_low.iloc[-1]
            current_atr = atr.iloc[-1] if not atr.empty else 0
            current_volume_ma = volume_ma.iloc[-1]

            # 檢查是否有有效突破
            upward_breakout = self._check_upward_breakout(
                data, current_price, current_channel_high, current_volume, current_volume_ma
            )

            downward_breakout = self._check_downward_breakout(
                data, current_price, current_channel_low, current_volume, current_volume_ma
            )

            # 趨勢確認
            trend_bullish = ma_fast.iloc[-1] > ma_slow.iloc[-1]
            trend_bearish = ma_fast.iloc[-1] < ma_slow.iloc[-1]

            # 向上突破買入信號
            if upward_breakout and trend_bullish:
                strength = self._calculate_breakout_strength(
                    current_price,
                    current_channel_high,
                    current_channel_low,
                    current_volume,
                    current_volume_ma,
                    current_atr,
                    "up",
                )

                signal = TradingSignal(
                    symbol=data.attrs.get("symbol", "UNKNOWN"),
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        "strategy": "breakout",
                        "breakout_type": "upward",
                        "channel_high": current_channel_high,
                        "channel_low": current_channel_low,
                        "atr": current_atr,
                        "volume_ratio": current_volume / current_volume_ma,
                        "stop_loss_price": current_price - (current_atr * self.atr_stop_multiplier),
                        "reason": "upward_breakout",
                    },
                )
                signals.append(signal)

            # 向下突破賣出信號
            elif downward_breakout and trend_bearish:
                strength = self._calculate_breakout_strength(
                    current_price,
                    current_channel_high,
                    current_channel_low,
                    current_volume,
                    current_volume_ma,
                    current_atr,
                    "down",
                )

                signal = TradingSignal(
                    symbol=data.attrs.get("symbol", "UNKNOWN"),
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        "strategy": "breakout",
                        "breakout_type": "downward",
                        "channel_high": current_channel_high,
                        "channel_low": current_channel_low,
                        "atr": current_atr,
                        "volume_ratio": current_volume / current_volume_ma,
                        "stop_loss_price": current_price + (current_atr * self.atr_stop_multiplier),
                        "reason": "downward_breakout",
                    },
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"{self.name}: Error calculating signals: {e}")

        return signals

    def _check_upward_breakout(
        self,
        data: pd.DataFrame,
        current_price: float,
        channel_high: float,
        current_volume: float,
        volume_ma: float,
    ) -> bool:
        """檢查向上突破條件"""
        # 價格突破通道上軌
        price_breakout = current_price > channel_high * (1 + self.breakout_threshold)

        # 成交量放大確認
        volume_confirmation = current_volume > volume_ma * self.volume_multiplier

        # 檢查是否經過充分整理
        recent_data = data.tail(self.consolidation_period)
        price_range = (recent_data["high"].max() - recent_data["low"].min()) / recent_data[
            "close"
        ].mean()
        consolidation = price_range < 0.05  # 5%以內整理

        return price_breakout and volume_confirmation and consolidation

    def _check_downward_breakout(
        self,
        data: pd.DataFrame,
        current_price: float,
        channel_low: float,
        current_volume: float,
        volume_ma: float,
    ) -> bool:
        """檢查向下突破條件"""
        # 價格跌破通道下軌
        price_breakout = current_price < channel_low * (1 - self.breakout_threshold)

        # 成交量放大確認
        volume_confirmation = current_volume > volume_ma * self.volume_multiplier

        # 檢查是否經過充分整理
        recent_data = data.tail(self.consolidation_period)
        price_range = (recent_data["high"].max() - recent_data["low"].min()) / recent_data[
            "close"
        ].mean()
        consolidation = price_range < 0.05  # 5%以內整理

        return price_breakout and volume_confirmation and consolidation

    def _calculate_breakout_strength(
        self,
        current_price: float,
        channel_high: float,
        channel_low: float,
        current_volume: float,
        volume_ma: float,
        atr: float,
        direction: str,
    ) -> float:
        """計算突破信號強度"""

        if direction == "up":
            # 突破幅度強度
            breakout_distance = (current_price - channel_high) / channel_high
            breakout_strength = min(breakout_distance / 0.02, 1.0)  # 2%完全突破

        else:  # down
            # 突破幅度強度
            breakout_distance = (channel_low - current_price) / channel_low
            breakout_strength = min(breakout_distance / 0.02, 1.0)

        # 成交量強度
        volume_ratio = current_volume / volume_ma
        volume_strength = min((volume_ratio - 1) / 2, 1.0)  # 3倍量能滿分

        # 通道寬度 (越窄突破越有效)
        channel_width = (channel_high - channel_low) / ((channel_high + channel_low) / 2)
        channel_strength = max(0, 1 - (channel_width / 0.1))  # 10%寬度基準

        # ATR標準化 (波動率考量)
        atr_normalized = min(atr / (current_price * 0.02), 1.0)
        atr_strength = 1 - atr_normalized  # 波動率越低突破越可靠

        # 綜合強度
        strength = (
            breakout_strength * 0.4
            + volume_strength * 0.3
            + channel_strength * 0.2
            + atr_strength * 0.1
        )

        return max(0.1, min(1.0, strength))

    def get_position_size(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> float:
        """
        基於ATR動態計算持倉大小

        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格

        Returns:
            持倉大小
        """
        # 獲取止損價位
        stop_loss_price = signal.metadata.get("stop_loss_price", current_price * 0.98)

        # 計算每股風險
        risk_per_share = abs(current_price - stop_loss_price)

        # 最大風險金額
        max_risk_amount = portfolio_value * self.max_risk_per_trade

        # 根據風險計算股數
        if risk_per_share > 0:
            max_shares_by_risk = max_risk_amount / risk_per_share
        else:
            max_shares_by_risk = float("inf")

        # 基礎持倉金額
        base_position_value = portfolio_value * self.position_size_pct
        base_shares = base_position_value / current_price

        # 取較小值確保風險控制
        shares = min(base_shares, max_shares_by_risk)

        # 根據信號強度調整
        strength_multiplier = 0.5 + (signal.strength * 0.5)
        shares *= strength_multiplier

        # 賣出信號返回負值
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            shares = -shares

        return shares

    def apply_risk_management(
        self, position: Position, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        突破策略風險管理 - ATR動態止損

        Args:
            position: 當前持倉
            market_data: 市場數據

        Returns:
            風險管理行動
        """
        action = {
            "action": "hold",
            "new_size": position.size,
            "reason": "No action needed",
            "stop_loss": None,
            "take_profit": None,
        }

        if market_data.empty or len(market_data) < self.atr_period:
            return action

        current_price = market_data["close"].iloc[-1]
        entry_price = position.avg_price

        # 計算當前ATR
        high_prices = market_data["high"]
        low_prices = market_data["low"]
        close_prices = market_data["close"]

        atr = self.indicator_calc.calculate_atr(
            high_prices, low_prices, close_prices, self.atr_period
        )
        current_atr = atr.iloc[-1] if not atr.empty else 0

        if position.size > 0:  # 多頭持倉
            # ATR動態止損
            stop_loss_price = current_price - (current_atr * self.atr_stop_multiplier)

            # 止盈目標
            take_profit_price = entry_price + (current_atr * self.profit_target_atr)

            # 檢查止損
            if current_price <= stop_loss_price:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"ATR stop loss triggered at {current_price:.2f}",
                        "stop_loss": stop_loss_price,
                    }
                )

            # 檢查止盈
            elif current_price >= take_profit_price:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Profit target reached at {current_price:.2f}",
                        "take_profit": take_profit_price,
                    }
                )

        elif position.size < 0:  # 空頭持倉
            # ATR動態止損
            stop_loss_price = current_price + (current_atr * self.atr_stop_multiplier)

            # 止盈目標
            take_profit_price = entry_price - (current_atr * self.profit_target_atr)

            # 檢查止損
            if current_price >= stop_loss_price:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"ATR stop loss triggered at {current_price:.2f}",
                        "stop_loss": stop_loss_price,
                    }
                )

            # 檢查止盈
            elif current_price <= take_profit_price:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Profit target reached at {current_price:.2f}",
                        "take_profit": take_profit_price,
                    }
                )

        return action

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """兼容性方法"""
        return self.calculate_signals(data)

    def calculate_position_size(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> float:
        """兼容性方法"""
        return self.get_position_size(signal, portfolio_value, current_price)

    def risk_management(self, position: Position, market_data: pd.DataFrame) -> Dict[str, Any]:
        """兼容性方法"""
        return self.apply_risk_management(position, market_data)


def create_breakout_strategy(
    symbols: List[str] = None, initial_capital: float = 100000
) -> BreakoutStrategy:
    """
    創建突破策略實例

    Args:
        symbols: 交易標的列表
        initial_capital: 初始資金

    Returns:
        突破策略實例
    """
    config = StrategyConfig(
        name="breakout_strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.025,  # 中等風險
        max_positions=6,
        symbols or [],
        parameters={
            "channel_period": 20,
            "breakout_threshold": 0.001,
            "volume_period": 20,
            "volume_multiplier": 1.5,
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
            "ma_fast": 10,
            "ma_slow": 20,
            "position_size_pct": 0.12,
            "max_risk_per_trade": 0.02,
            "profit_target_atr": 3.0,
            "min_breakout_bars": 3,
            "consolidation_period": 10,
        },
    )

    return BreakoutStrategy(config, initial_capital)
