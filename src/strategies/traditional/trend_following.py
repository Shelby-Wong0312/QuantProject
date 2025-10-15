"""
趨勢跟隨策略 - 基於多時間框架趨勢分析
使用移動平均、MACD、ADX趨勢強度確認
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


class TrendFollowingStrategy(BaseStrategy):
    """
    趨勢跟隨策略 - 順勢而為，持有強勢趨勢

    核心邏輯：
    - 多重均線排列 + ADX強勢 + MACD同向 = 入場
    - 趨勢減弱或反轉 = 出場
    - 金字塔加倉，追求大趨勢利潤
    """

    def _initialize_parameters(self) -> None:
        """初始化策略參數"""
        params = self.config.parameters

        # 移動平均參數
        self.ma_short = params.get("ma_short", 10)
        self.ma_medium = params.get("ma_medium", 20)
        self.ma_long = params.get("ma_long", 50)
        self.ma_trend = params.get("ma_trend", 200)

        # MACD參數
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal = params.get("macd_signal", 9)

        # ADX參數
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25)

        # 趨勢確認參數
        self.trend_strength_period = params.get("trend_strength_period", 20)
        self.min_trend_bars = params.get("min_trend_bars", 5)

        # 風險參數
        self.initial_position_pct = params.get("initial_position_pct", 0.08)
        self.pyramid_add_pct = params.get("pyramid_add_pct", 0.04)
        self.max_pyramid_levels = params.get("max_pyramid_levels", 3)
        self.trailing_stop_atr = params.get("trailing_stop_atr", 2.0)

        # ATR參數
        self.atr_period = params.get("atr_period", 14)

        self.indicator_calc = IndicatorCalculator()

        logger.info(
            f"{self.name}: Initialized with MA({self.ma_short}/{self.ma_medium}/"
            f"{self.ma_long}/{self.ma_trend}), ADX({self.adx_period})"
        )

    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        計算趨勢跟隨交易信號

        Args:
            data: OHLCV數據

        Returns:
            交易信號列表
        """
        signals = []

        if len(data) < max(self.ma_trend, self.trend_strength_period) + 10:
            return signals

        try:
            # 計算技術指標
            close_prices = data["close"]
            high_prices = data["high"]
            low_prices = data["low"]

            # 移動平均線
            ma_short = close_prices.rolling(window=self.ma_short).mean()
            ma_medium = close_prices.rolling(window=self.ma_medium).mean()
            ma_long = close_prices.rolling(window=self.ma_long).mean()
            ma_trend = close_prices.rolling(window=self.ma_trend).mean()

            # MACD
            macd_data = self.indicator_calc.calculate_macd(
                close_prices, self.macd_fast, self.macd_slow, self.macd_signal
            )

            # ADX (趨勢強度)
            adx = self.indicator_calc.calculate_adx(
                high_prices, low_prices, close_prices, self.adx_period
            )

            # ATR
            atr = self.indicator_calc.calculate_atr(
                high_prices, low_prices, close_prices, self.atr_period
            )

            # 最新值
            current_price = close_prices.iloc[-1]
            current_ma_short = ma_short.iloc[-1]
            current_ma_medium = ma_medium.iloc[-1]
            current_ma_long = ma_long.iloc[-1]
            current_ma_trend = ma_trend.iloc[-1]

            current_macd = macd_data["MACD"].iloc[-1] if not macd_data.empty else 0
            current_signal = macd_data["Signal"].iloc[-1] if not macd_data.empty else 0
            current_adx = adx.iloc[-1] if not adx.empty else 0
            current_atr = atr.iloc[-1] if not atr.empty else 0

            # 趨勢方向分析
            trend_direction = self._analyze_trend_direction(
                current_price,
                current_ma_short,
                current_ma_medium,
                current_ma_long,
                current_ma_trend,
            )

            # 趨勢強度確認
            trend_strength = self._calculate_trend_strength(data, ma_short, ma_medium, ma_long, adx)

            # 多頭趨勢信號
            if (
                trend_direction == "bullish"
                and current_adx > self.adx_threshold
                and current_macd > current_signal
                and trend_strength > 0.6
            ):

                strength = self._calculate_trend_signal_strength(
                    trend_direction, trend_strength, current_adx, current_macd, current_signal
                )

                signal = TradingSignal(
                    symbol=data.attrs.get("symbol", "UNKNOWN"),
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        "strategy": "trend_following",
                        "trend_direction": trend_direction,
                        "trend_strength": trend_strength,
                        "adx": current_adx,
                        "macd": current_macd,
                        "atr": current_atr,
                        "trailing_stop": current_price - (current_atr * self.trailing_stop_atr),
                        "reason": "bullish_trend_confirmed",
                    },
                )
                signals.append(signal)

            # 空頭趨勢信號
            elif (
                trend_direction == "bearish"
                and current_adx > self.adx_threshold
                and current_macd < current_signal
                and trend_strength > 0.6
            ):

                strength = self._calculate_trend_signal_strength(
                    trend_direction, trend_strength, current_adx, current_macd, current_signal
                )

                signal = TradingSignal(
                    symbol=data.attrs.get("symbol", "UNKNOWN"),
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        "strategy": "trend_following",
                        "trend_direction": trend_direction,
                        "trend_strength": trend_strength,
                        "adx": current_adx,
                        "macd": current_macd,
                        "atr": current_atr,
                        "trailing_stop": current_price + (current_atr * self.trailing_stop_atr),
                        "reason": "bearish_trend_confirmed",
                    },
                )
                signals.append(signal)

            # 加倉信號 (已有持倉時)
            pyramid_signal = self._check_pyramid_opportunity(
                data, trend_direction, trend_strength, current_adx
            )
            if pyramid_signal:
                signals.append(pyramid_signal)

        except Exception as e:
            logger.error(f"{self.name}: Error calculating signals: {e}")

        return signals

    def _analyze_trend_direction(
        self, price: float, ma_short: float, ma_medium: float, ma_long: float, ma_trend: float
    ) -> str:
        """分析趨勢方向"""
        # 均線排列檢查
        bullish_alignment = price > ma_short > ma_medium > ma_long > ma_trend
        bearish_alignment = price < ma_short < ma_medium < ma_long < ma_trend

        if bullish_alignment:
            return "bullish"
        elif bearish_alignment:
            return "bearish"
        else:
            return "sideways"

    def _calculate_trend_strength(
        self,
        data: pd.DataFrame,
        ma_short: pd.Series,
        ma_medium: pd.Series,
        ma_long: pd.Series,
        adx: pd.Series,
    ) -> float:
        """計算趨勢強度"""
        try:
            # 均線斜率強度
            short_slope = (ma_short.iloc[-1] - ma_short.iloc[-5]) / ma_short.iloc[-5]
            medium_slope = (ma_medium.iloc[-1] - ma_medium.iloc[-10]) / ma_medium.iloc[-10]
            long_slope = (ma_long.iloc[-1] - ma_long.iloc[-20]) / ma_long.iloc[-20]

            # 斜率一致性
            slope_consistency = 0
            if short_slope > 0 and medium_slope > 0 and long_slope > 0:
                slope_consistency = 1.0
            elif short_slope < 0 and medium_slope < 0 and long_slope < 0:
                slope_consistency = 1.0
            else:
                slope_consistency = 0.5

            # ADX強度標準化
            adx_strength = min(adx.iloc[-1] / 50, 1.0) if not adx.empty else 0

            # 價格動量
            recent_returns = data["close"].pct_change().tail(10)
            momentum_consistency = len(recent_returns[recent_returns > 0]) / len(recent_returns)
            if recent_returns.mean() < 0:
                momentum_consistency = 1 - momentum_consistency

            # 綜合強度
            strength = slope_consistency * 0.4 + adx_strength * 0.4 + momentum_consistency * 0.2

            return max(0.1, min(1.0, strength))

        except:
            return 0.5

    def _calculate_trend_signal_strength(
        self, direction: str, trend_strength: float, adx: float, macd: float, signal: float
    ) -> float:
        """計算趨勢信號強度"""

        # 基礎趨勢強度
        base_strength = trend_strength

        # ADX強度 (25-50範圍標準化)
        adx_strength = min((adx - 25) / 25, 1.0) if adx > 25 else 0

        # MACD強度
        macd_diff = abs(macd - signal)
        macd_strength = min(macd_diff / abs(signal) if signal != 0 else 0.5, 1.0)

        # 綜合強度
        strength = base_strength * 0.5 + adx_strength * 0.3 + macd_strength * 0.2

        return max(0.1, min(1.0, strength))

    def _check_pyramid_opportunity(
        self, data: pd.DataFrame, trend_direction: str, trend_strength: float, adx: float
    ) -> TradingSignal:
        """檢查金字塔加倉機會"""
        # 簡化實現 - 實際應檢查現有持倉
        if trend_strength > 0.8 and adx > 30:
            # 這裡應該檢查現有持倉並決定是否加倉
            # 為了簡化，暫時返回None
            pass
        return None

    def get_position_size(
        self, signal: TradingSignal, portfolio_value: float, current_price: float
    ) -> float:
        """
        趨勢跟隨策略持倉計算 - 支持金字塔加倉

        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格

        Returns:
            持倉大小
        """
        # 檢查是否是加倉信號
        is_pyramid = signal.metadata.get("is_pyramid", False)

        if is_pyramid:
            # 加倉持倉較小
            position_value = portfolio_value * self.pyramid_add_pct
        else:
            # 初始持倉
            position_value = portfolio_value * self.initial_position_pct

        # 根據趨勢強度調整
        trend_strength = signal.metadata.get("trend_strength", 0.5)
        strength_multiplier = 0.5 + (trend_strength * 0.5)

        # 根據ADX調整 (趨勢越強持倉越大)
        adx = signal.metadata.get("adx", 25)
        adx_multiplier = min(adx / 50, 1.2)  # 最多120%

        # 計算股數
        adjusted_value = position_value * strength_multiplier * adx_multiplier
        shares = adjusted_value / current_price

        # 賣出信號返回負值
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            shares = -shares

        return shares

    def apply_risk_management(
        self, position: Position, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        趨勢跟隨風險管理 - 動態追蹤止損

        Args:
            position: 當前持倉
            market_data: 市場數據

        Returns:
            風險管理行動
        """
        action = {
            "action": "hold",
            "new_size": position.size,
            "reason": "Trend continues",
            "stop_loss": None,
            "take_profit": None,
        }

        if market_data.empty or len(market_data) < self.atr_period:
            return action

        current_price = market_data["close"].iloc[-1]

        # 計算ATR動態止損
        high_prices = market_data["high"]
        low_prices = market_data["low"]
        close_prices = market_data["close"]

        atr = self.indicator_calc.calculate_atr(
            high_prices, low_prices, close_prices, self.atr_period
        )
        current_atr = atr.iloc[-1] if not atr.empty else 0

        # 趨勢檢查
        ma_short = close_prices.rolling(window=self.ma_short).mean()
        ma_medium = close_prices.rolling(window=self.ma_medium).mean()

        trend_intact = True

        if position.size > 0:  # 多頭持倉
            # 動態追蹤止損
            trailing_stop = current_price - (current_atr * self.trailing_stop_atr)

            # 更新止損位 (只能上移)
            if hasattr(position, "trailing_stop"):
                position.trailing_stop = max(position.trailing_stop, trailing_stop)
            else:
                position.trailing_stop = trailing_stop

            # 檢查止損觸發
            if current_price <= position.trailing_stop:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Trailing stop triggered at {current_price:.2f}",
                        "stop_loss": position.trailing_stop,
                    }
                )

            # 檢查趨勢完整性
            elif current_price < ma_short.iloc[-1] and ma_short.iloc[-1] < ma_medium.iloc[-1]:
                trend_intact = False

        elif position.size < 0:  # 空頭持倉
            # 動態追蹤止損
            trailing_stop = current_price + (current_atr * self.trailing_stop_atr)

            # 更新止損位 (只能下移)
            if hasattr(position, "trailing_stop"):
                position.trailing_stop = min(position.trailing_stop, trailing_stop)
            else:
                position.trailing_stop = trailing_stop

            # 檢查止損觸發
            if current_price >= position.trailing_stop:
                action.update(
                    {
                        "action": "close",
                        "new_size": 0,
                        "reason": f"Trailing stop triggered at {current_price:.2f}",
                        "stop_loss": position.trailing_stop,
                    }
                )

            # 檢查趨勢完整性
            elif current_price > ma_short.iloc[-1] and ma_short.iloc[-1] > ma_medium.iloc[-1]:
                trend_intact = False

        # 趨勢反轉檢查
        if not trend_intact:
            action.update({"action": "close", "new_size": 0, "reason": "Trend reversal detected"})

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


def create_trend_following_strategy(
    symbols: List[str] = None, initial_capital: float = 100000
) -> TrendFollowingStrategy:
    """
    創建趨勢跟隨策略實例

    Args:
        symbols: 交易標的列表
        initial_capital: 初始資金

    Returns:
        趨勢跟隨策略實例
    """
    config = StrategyConfig(
        name="trend_following_strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.03,  # 較高風險承受
        max_positions=4,  # 集中持倉
        symbols=symbols or [],
        parameters={
            "ma_short": 10,
            "ma_medium": 20,
            "ma_long": 50,
            "ma_trend": 200,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 25,
            "trend_strength_period": 20,
            "min_trend_bars": 5,
            "initial_position_pct": 0.08,
            "pyramid_add_pct": 0.04,
            "max_pyramid_levels": 3,
            "trailing_stop_atr": 2.0,
            "atr_period": 14,
        },
    )

    return TrendFollowingStrategy(config, initial_capital)
