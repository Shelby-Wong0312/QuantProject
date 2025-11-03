"""
動量策略 - 基於價格動量和成交量的策略
使用RSI、MACD、成交量分析
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from ..base_strategy import BaseStrategy
from ..strategy_interface import TradingSignal, SignalType, StrategyConfig, Position
from ...indicators.momentum_indicators import RSI, MACD
from ...indicators.volatility_indicators import ATR

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    動量策略 - 追隨價格和成交量動量

    核心邏輯：
    - RSI > 60 且 MACD 金叉 = 買入信號
    - RSI < 40 且 MACD 死叉 = 賣出信號
    - 成交量確認動量強度
    """

    def _initialize_parameters(self) -> None:
        """初始化策略參數"""
        params = self.config.parameters

        # RSI參數
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_buy_threshold = params.get("rsi_buy_threshold", 60)
        self.rsi_sell_threshold = params.get("rsi_sell_threshold", 40)

        # MACD參數
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal = params.get("macd_signal", 9)

        # 成交量參數
        self.volume_period = params.get("volume_period", 20)
        self.volume_threshold = params.get("volume_threshold", 1.5)  # 倍數

        # 風險參數
        self.stop_loss_pct = params.get("stop_loss_pct", 0.02)
        self.take_profit_pct = params.get("take_profit_pct", 0.04)
        self.position_size_pct = params.get("position_size_pct", 0.1)

        # 初始化指標實例
        self.rsi_indicator = RSI(period=self.rsi_period)
        self.macd_indicator = MACD(
            fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal
        )

        logger.info(
            f"{self.name}: Initialized with RSI({self.rsi_period}), "
            f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})"
        )

    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        計算動量交易信號

        Args:
            data: OHLCV數據

        Returns:
            交易信號列表
        """
        []

        if len(data) < max(self.macd_slow, self.volume_period, self.rsi_period) + 10:
            return signals

        try:
            # 計算技術指標
            rsi = self.rsi_indicator.calculate(data)
            macd_data = self.macd_indicator.calculate(data)

            # 成交量分析
            volume_sma = data["volume"].rolling(window=self.volume_period).mean()
            volume_ratio = data["volume"] / volume_sma

            # 最新指標值
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            current_macd = macd_data["macd"].iloc[-1] if not macd_data.empty else 0
            current_signal = macd_data["signal"].iloc[-1] if not macd_data.empty else 0
            current_volume_ratio = volume_ratio.iloc[-1] if not volume_ratio.empty else 1

            # 檢測MACD金叉死叉
            macd_bullish = self._detect_macd_crossover(macd_data, "bullish")
            macd_bearish = self._detect_macd_crossover(macd_data, "bearish")

            # 動量買入信號
            if (
                current_rsi > self.rsi_buy_threshold
                and macd_bullish
                and current_volume_ratio > self.volume_threshold
            ):

                strength = self._calculate_signal_strength(
                    current_rsi, current_macd, current_signal, current_volume_ratio, "buy"
                )

                signal = TradingSignal(
                    symbol=data.attrs.get("symbol", "UNKNOWN"),
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=data["close"].iloc[-1],
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        "strategy": "momentum",
                        "rsi": current_rsi,
                        "macd": current_macd,
                        "volume_ratio": current_volume_ratio,
                        "reason": "momentum_bullish",
                    },
                )
                signals.append(signal)

            # 動量賣出信號
            elif (
                current_rsi < self.rsi_sell_threshold
                and macd_bearish
                and current_volume_ratio > self.volume_threshold
            ):

                strength = self._calculate_signal_strength(
                    current_rsi, current_macd, current_signal, current_volume_ratio, "sell"
                )

                signal = TradingSignal(
                    symbol=data.attrs.get("symbol", "UNKNOWN"),
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=data["close"].iloc[-1],
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        "strategy": "momentum",
                        "rsi": current_rsi,
                        "macd": current_macd,
                        "volume_ratio": current_volume_ratio,
                        "reason": "momentum_bearish",
                    },
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"{self.name}: Error calculating signals: {e}")

        return signals

    def _detect_macd_crossover(self, macd_data: pd.DataFrame, direction: str) -> bool:
        """檢測MACD金叉死叉"""
        if len(macd_data) < 3:
            return False

        macd = macd_data["macd"].iloc[-3:]
        signal = macd_data["signal"].iloc[-3:]

        if direction == "bullish":
            # 金叉：MACD從下方穿越信號線
            return macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]
        else:
            # 死叉：MACD從上方穿越信號線
            return macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]

    def _calculate_signal_strength(
        self, rsi: float, macd: float, signal: float, volume_ratio: float, direction: str
    ) -> float:
        """計算信號強度"""
        strength = 0.0

        if direction == "buy":
            # RSI強度 (60-100範圍)
            rsi_strength = min((rsi - 60) / 40, 1.0)

            # MACD強度
            macd_strength = min(abs(macd - signal) / abs(signal) if signal != 0 else 0.5, 1.0)

        else:  # sell
            # RSI強度 (0-40範圍)
            rsi_strength = min((40 - rsi) / 40, 1.0)

            # MACD強度
            macd_strength = min(abs(macd - signal) / abs(signal) if signal != 0 else 0.5, 1.0)

        # 成交量強度
        volume_strength = min((volume_ratio - 1) / 2, 1.0)

        # 綜合強度
        strength = rsi_strength * 0.4 + macd_strength * 0.4 + volume_strength * 0.2

        return max(0.1, min(1.0, strength))

    def get_position_size(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> float:
        """
        計算持倉大小

        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格

        Returns:
            持倉大小
        """
        # 基礎持倉比例
        base_position_value = portfolio_value * self.position_size_pct

        # 根據信號強度調整
        strength_multiplier = 0.5 + (signal.strength * 0.5)  # 0.5-1.0

        # 計算股數
        position_value = base_position_value * strength_multiplier
        shares = position_value / current_price

        # 賣出信號返回負值
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            shares = -shares

        return shares

    def apply_risk_management(
        self, position: Position, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        應用風險管理

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

        if market_data.empty:
            return action

        current_price = market_data["close"].iloc[-1]
        entry_price = position.entry_price

        # 計算收益率
        if position.size > 0:  # 多頭
            return_pct = (current_price - entry_price) / entry_price

            # 止損
            if return_pct < -self.stop_loss_pct:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Stop loss triggered: {return_pct:.2%}",
                        "stop_loss": entry_price * (1 - self.stop_loss_pct),
                    }
                )

            # 止盈
            elif return_pct > self.take_profit_pct:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Take profit triggered: {return_pct:.2%}",
                        "take_profit": entry_price * (1 + self.take_profit_pct),
                    }
                )

        elif position.size < 0:  # 空頭
            return_pct = (entry_price - current_price) / entry_price

            # 止損
            if return_pct < -self.stop_loss_pct:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Stop loss triggered: {return_pct:.2%}",
                        "stop_loss": entry_price * (1 + self.stop_loss_pct),
                    }
                )

            # 止盈
            elif return_pct > self.take_profit_pct:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Take profit triggered: {return_pct:.2%}",
                        "take_profit": entry_price * (1 - self.take_profit_pct),
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


def create_momentum_strategy(
    symbols: List[str] = None, initial_capital: float = 100000
) -> MomentumStrategy:
    """
    創建動量策略實例

    Args:
        symbols: 交易標的列表
        initial_capital: 初始資金

    Returns:
        動量策略實例
    """
    config = StrategyConfig(
        name="momentum_strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.02,
        max_positions=5,
        symbols or [],
        parameters={
            "rsi_period": 14,
            "rsi_buy_threshold": 60,
            "rsi_sell_threshold": 40,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "volume_period": 20,
            "volume_threshold": 1.5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "position_size_pct": 0.1,
        },
    )

    return MomentumStrategy(config, initial_capital)
