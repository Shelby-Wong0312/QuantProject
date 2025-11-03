"""
Technical Indicators Calculation Module
"""

import pandas as pd
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate various technical indicators for financial data
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average

        Args:
            data: Price series
            period: Number of periods

        Returns:
            SMA series
        """
        return data.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average

        Args:
            data: Price series
            period: Number of periods

        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index

        Args:
            data: Price series
            period: Number of periods (default: 14)

        Returns:
            RSI series (0-100)
        """
        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero
        rsi = rsi.fillna(50)

        return rsi

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence

        Args:
            data: Price series
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Args:
            data: Price series
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2)

        Returns:
            Tuple of (Upper band, Middle band (SMA), Lower band)
        """
        # Middle band (SMA)
        middle = TechnicalIndicators.sma(data, period)

        # Standard deviation
        std = data.rolling(window=period, min_periods=1).std()

        # Upper and lower bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average True Range

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default: 14)

        Returns:
            ATR series
        """
        # Calculate True Range components
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period (default: 14)
            smooth_k: %K smoothing period (default: 3)
            smooth_d: %D smoothing period (default: 3)

        Returns:
            Tuple of (%K, %D)
        """
        # Calculate rolling high and low
        rolling_high = high.rolling(window=period, min_periods=1).max()
        rolling_low = low.rolling(window=period, min_periods=1).min()

        # Calculate %K
        k_percent = 100 * ((close - rolling_low) / (rolling_high - rolling_low))

        # Smooth %K
        k_percent = k_percent.rolling(window=smooth_k, min_periods=1).mean()

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=smooth_d, min_periods=1).mean()

        return k_percent, d_percent

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average Directional Index

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period (default: 14)

        Returns:
            ADX series
        """
        # Calculate True Range
        atr = TechnicalIndicators.atr(high, low, close, period)

        # Calculate directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        # Positive directional movement
        pos_dm = pd.Series(0, index=high.index)
        pos_dm[(up_move > down_move) & (up_move > 0)] = up_move

        # Negative directional movement
        neg_dm = pd.Series(0, index=low.index)
        neg_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smooth the directional movements
        pos_di = 100 * (
            pos_dm.rolling(window=period).sum() / atr.rolling(window=period).sum()
        )
        neg_di = 100 * (
            neg_dm.rolling(window=period).sum() / atr.rolling(window=period).sum()
        )

        # Calculate DX
        dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di))

        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            OBV series
        """
        # Calculate price direction
        price_diff = close.diff()

        # Calculate OBV
        obv = pd.Series(0, index=close.index)
        obv[price_diff > 0] = volume[price_diff > 0]
        obv[price_diff < 0] = -volume[price_diff < 0]

        # Cumulative sum
        obv = obv.cumsum()

        return obv

    @staticmethod
    def vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series

        Returns:
            VWAP series
        """
        # Typical price
        typical_price = (high + low + close) / 3

        # Calculate VWAP
        cumulative_price_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

        vwap = cumulative_price_volume / cumulative_volume

        return vwap

    @staticmethod
    def calculate_all_indicators(
        df: pd.DataFrame, indicators: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Calculate all or specified indicators for a DataFrame

        Args:
            df: DataFrame with OHLCV columns
            indicators: List of indicators to calculate (None for all)

        Returns:
            DataFrame with additional indicator columns
        """
        result = df.copy()

        # Default to all indicators if none specified
        if indicators is None:
            indicators = [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bollinger",
                "atr",
                "stochastic",
                "adx",
                "obv",
                "vwap",
            ]

        # Ensure column names are lowercase
        result.columns = result.columns.str.lower()

        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(result.columns)
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return result

        # Calculate indicators
        if "sma" in indicators:
            result["sma_20"] = TechnicalIndicators.sma(result["close"], 20)
            result["sma_50"] = TechnicalIndicators.sma(result["close"], 50)

        if "ema" in indicators:
            result["ema_12"] = TechnicalIndicators.ema(result["close"], 12)
            result["ema_26"] = TechnicalIndicators.ema(result["close"], 26)

        if "rsi" in indicators:
            result["rsi"] = TechnicalIndicators.rsi(result["close"])

        if "macd" in indicators:
            macd, signal, histogram = TechnicalIndicators.macd(result["close"])
            result["macd"] = macd
            result["macd_signal"] = signal
            result["macd_histogram"] = histogram

        if "bollinger" in indicators:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(result["close"])
            result["bb_upper"] = upper
            result["bb_middle"] = middle
            result["bb_lower"] = lower

        if "atr" in indicators:
            result["atr"] = TechnicalIndicators.atr(
                result["high"], result["low"], result["close"]
            )

        if "stochastic" in indicators:
            k, d = TechnicalIndicators.stochastic(
                result["high"], result["low"], result["close"]
            )
            result["stoch_k"] = k
            result["stoch_d"] = d

        if "adx" in indicators:
            result["adx"] = TechnicalIndicators.adx(
                result["high"], result["low"], result["close"]
            )

        if "obv" in indicators:
            result["obv"] = TechnicalIndicators.obv(result["close"], result["volume"])

        if "vwap" in indicators:
            result["vwap"] = TechnicalIndicators.vwap(
                result["high"], result["low"], result["close"], result["volume"]
            )

        return result
