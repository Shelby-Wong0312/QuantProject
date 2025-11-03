"""
Volume Indicators - OBV, Volume SMA, MFI, A/D Line
"""

import pandas as pd
from .base_indicator import BaseIndicator
import logging

logger = logging.getLogger(__name__)


class OBV(BaseIndicator):
    """On-Balance Volume - Cumulative volume flow indicator"""

    def __init__(self, use_cache: bool = True):
        super().__init__(period=1, use_cache=use_cache)  # OBV doesn't use period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        if not self.validate(data):
            return pd.Series()

        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Calculate price direction
        price_diff = data["close"].diff()

        # Calculate OBV
        obv = pd.Series(0, index=data.index, dtype=float)
        obv[price_diff > 0] = data["volume"][price_diff > 0]
        obv[price_diff < 0] = -data["volume"][price_diff < 0]
        obv = obv.cumsum()

        self._save_to_cache(cache_key, obv)
        return obv

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate OBV divergence signals"""
        obv = self.calculate(data)
        pd.DataFrame(index=data.index)

        signals["obv"] = obv
        signals["obv_sma"] = obv.rolling(window=20).mean()

        # OBV trend
        signals["obv_trend_up"] = obv > signals["obv_sma"]
        signals["obv_trend_down"] = obv < signals["obv_sma"]

        # Divergence detection (simplified)
        price_higher = data["close"] > data["close"].shift(20)
        obv_lower = obv < obv.shift(20)
        signals["bearish_divergence"] = price_higher & obv_lower

        price_lower = data["close"] < data["close"].shift(20)
        obv_higher = obv > obv.shift(20)
        signals["bullish_divergence"] = price_lower & obv_higher

        # Buy/Sell signals based on OBV breakouts
        obv_high_20 = obv.rolling(window=20).max()
        obv_low_20 = obv.rolling(window=20).min()

        signals["buy"] = (obv > obv_high_20.shift(1)) & signals["obv_trend_up"]
        signals["sell"] = (obv < obv_low_20.shift(1)) & signals["obv_trend_down"]
        signals["hold"] = ~(signals["buy"] | signals["sell"])

        return signals


class VolumeSMA(BaseIndicator):
    """Volume Simple Moving Average with anomaly detection"""

    def __init__(self, period: int = 20, use_cache: bool = True):
        super().__init__(period, use_cache)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume SMA and volume metrics"""
        if not self.validate(data):
            return pd.DataFrame()

        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Calculate Volume SMA
        volume_sma = data["volume"].rolling(window=self.period).mean()

        # Calculate Volume Standard Deviation
        volume_std = data["volume"].rolling(window=self.period).std()

        # Volume Ratio (current volume / average volume)
        volume_ratio = data["volume"] / volume_sma

        result = pd.DataFrame(
            {
                "volume_sma": volume_sma,
                "volume_std": volume_std,
                "volume_ratio": volume_ratio,
                "volume_zscore": (data["volume"] - volume_sma) / volume_std,
            },
            index=data.index,
        )

        self._save_to_cache(cache_key, result)
        return result

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume anomaly signals"""
        vol_data = self.calculate(data)
        pd.DataFrame(index=data.index)

        signals["volume"] = data["volume"]
        signals["volume_sma"] = vol_data["volume_sma"]
        signals["volume_ratio"] = vol_data["volume_ratio"]

        # High volume detection (2x average)
        signals["high_volume"] = vol_data["volume_ratio"] > 2.0
        signals["extreme_volume"] = vol_data["volume_ratio"] > 3.0

        # Low volume detection
        signals["low_volume"] = vol_data["volume_ratio"] < 0.5

        # Volume surge with price movement
        price_up = data["close"] > data["close"].shift(1)
        price_down = data["close"] < data["close"].shift(1)

        signals["volume_surge_up"] = signals["high_volume"] & price_up
        signals["volume_surge_down"] = signals["high_volume"] & price_down

        # Buy on volume surge with price increase
        signals["buy"] = signals["volume_surge_up"] & (vol_data["volume_zscore"] > 2)

        # Sell on volume surge with price decrease
        signals["sell"] = signals["volume_surge_down"] & (vol_data["volume_zscore"] > 2)

        signals["hold"] = ~(signals["buy"] | signals["sell"])

        return signals


class MFI(BaseIndicator):
    """Money Flow Index - Volume-weighted RSI"""

    def __init__(self, period: int = 14, use_cache: bool = True):
        super().__init__(period, use_cache)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Money Flow Index"""
        if not self.validate(data):
            return pd.Series()

        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Calculate Typical Price
        typical_price = (data["high"] + data["low"] + data["close"]) / 3

        # Calculate Raw Money Flow
        raw_money_flow = typical_price * data["volume"]

        # Calculate Positive and Negative Money Flow
        money_flow_positive = pd.Series(0, index=data.index)
        money_flow_negative = pd.Series(0, index=data.index)

        # Compare with previous typical price
        price_diff = typical_price.diff()
        money_flow_positive[price_diff > 0] = raw_money_flow[price_diff > 0]
        money_flow_negative[price_diff < 0] = raw_money_flow[price_diff < 0]

        # Calculate Money Flow Ratio
        positive_flow = money_flow_positive.rolling(window=self.period).sum()
        negative_flow = money_flow_negative.rolling(window=self.period).sum()

        # Avoid division by zero
        money_flow_ratio = positive_flow / negative_flow.replace(0, 1)

        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))

        self._save_to_cache(cache_key, mfi)
        return mfi

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MFI overbought/oversold signals"""
        mfi = self.calculate(data)
        pd.DataFrame(index=data.index)

        signals["mfi"] = mfi

        # Overbought/Oversold levels
        signals["overbought"] = mfi > 80
        signals["oversold"] = mfi < 20

        # Buy signal when MFI crosses above 20 (exiting oversold)
        signals["buy"] = (mfi > 20) & (mfi.shift(1) <= 20)

        # Sell signal when MFI crosses below 80 (exiting overbought)
        signals["sell"] = (mfi < 80) & (mfi.shift(1) >= 80)

        signals["hold"] = ~(signals["buy"] | signals["sell"])

        # Divergence detection
        price_higher = data["close"] > data["close"].rolling(20).max().shift(1)
        mfi_lower = mfi < mfi.rolling(20).max().shift(1)
        signals["bearish_divergence"] = price_higher & mfi_lower & signals["overbought"]

        price_lower = data["close"] < data["close"].rolling(20).min().shift(1)
        mfi_higher = mfi > mfi.rolling(20).min().shift(1)
        signals["bullish_divergence"] = price_lower & mfi_higher & signals["oversold"]

        return signals


class ADLine(BaseIndicator):
    """Accumulation/Distribution Line - Volume and price relationship"""

    def __init__(self, use_cache: bool = True):
        super().__init__(period=1, use_cache=use_cache)  # A/D Line doesn't use period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        if not self.validate(data):
            return pd.Series()

        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Calculate Money Flow Multiplier
        clv = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (
            data["high"] - data["low"]
        )
        clv = clv.fillna(0)  # Handle division by zero when high == low

        # Calculate Money Flow Volume
        money_flow_volume = clv * data["volume"]

        # Calculate A/D Line (cumulative sum)
        ad_line = money_flow_volume.cumsum()

        self._save_to_cache(cache_key, ad_line)
        return ad_line

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate A/D Line trend and divergence signals"""
        ad_line = self.calculate(data)
        pd.DataFrame(index=data.index)

        signals["ad_line"] = ad_line
        signals["ad_line_sma"] = ad_line.rolling(window=20).mean()

        # Trend detection
        signals["ad_trend_up"] = ad_line > signals["ad_line_sma"]
        signals["ad_trend_down"] = ad_line < signals["ad_line_sma"]

        # Breakout detection
        ad_high_20 = ad_line.rolling(window=20).max()
        ad_low_20 = ad_line.rolling(window=20).min()

        signals["ad_breakout_up"] = ad_line > ad_high_20.shift(1)
        signals["ad_breakout_down"] = ad_line < ad_low_20.shift(1)

        # Divergence with price
        price_higher = data["close"] > data["close"].rolling(20).max().shift(1)
        ad_lower = ad_line < ad_line.rolling(20).max().shift(1)
        signals["bearish_divergence"] = price_higher & ad_lower

        price_lower = data["close"] < data["close"].rolling(20).min().shift(1)
        ad_higher = ad_line > ad_line.rolling(20).min().shift(1)
        signals["bullish_divergence"] = price_lower & ad_higher

        # Buy/Sell signals
        signals["buy"] = signals["ad_breakout_up"] & signals["ad_trend_up"]
        signals["sell"] = signals["ad_breakout_down"] & signals["ad_trend_down"]
        signals["hold"] = ~(signals["buy"] | signals["sell"])

        # Chaikin Oscillator (3-day EMA - 10-day EMA of A/D Line)
        ad_ema3 = ad_line.ewm(span=3, adjust=False).mean()
        ad_ema10 = ad_line.ewm(span=10, adjust=False).mean()
        signals["chaikin_oscillator"] = ad_ema3 - ad_ema10

        return signals
