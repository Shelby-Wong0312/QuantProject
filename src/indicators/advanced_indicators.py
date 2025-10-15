"""
Advanced Technical Indicators
進階技術指標計算
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class AdvancedIndicators:
    """進階技術指標計算器"""

    def calculate_all(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        計算所有技術指標

        Args:
            df: OHLCV 數據

        Returns:
            指標字典
        """
        indicators = {}

        # CCI
        indicators["CCI_20"] = self.calculate_cci(df, 20)

        # RSI
        indicators["RSI_14"] = self.calculate_rsi(df["close"], 14)

        # MACD
        macd_result = self.calculate_macd(df["close"])
        indicators["MACD"] = macd_result["macd"]
        indicators["MACD_Signal"] = macd_result["signal"]
        indicators["MACD_Histogram"] = macd_result["histogram"]

        # Bollinger Bands
        bb_result = self.calculate_bollinger_bands(df["close"])
        indicators["BB_Upper"] = bb_result["upper"]
        indicators["BB_Lower"] = bb_result["lower"]
        indicators["BB_Middle"] = bb_result["middle"]

        # Stochastic
        stoch_result = self.calculate_stochastic(df)
        indicators["Stochastic_K"] = stoch_result["k"]
        indicators["Stochastic_D"] = stoch_result["d"]

        # Volume indicators
        indicators["OBV"] = self.calculate_obv(df)
        indicators["Volume_SMA"] = df["volume"].rolling(20).mean().iloc[-1]

        return indicators

    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> float:
        """計算 CCI"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(period).mean()
        mean_deviation = np.abs(typical_price - sma_tp).rolling(period).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci.iloc[-1] if len(cci) > 0 else 0

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """計算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict:
        """計算 MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal

        return {
            "macd": macd.iloc[-1] if len(macd) > 0 else 0,
            "signal": macd_signal.iloc[-1] if len(macd_signal) > 0 else 0,
            "histogram": macd_histogram.iloc[-1] if len(macd_histogram) > 0 else 0,
        }

    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Dict:
        """計算布林帶"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return {
            "upper": upper.iloc[-1] if len(upper) > 0 else 0,
            "middle": sma.iloc[-1] if len(sma) > 0 else 0,
            "lower": lower.iloc[-1] if len(lower) > 0 else 0,
        }

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """計算隨機指標"""
        low_min = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        k = 100 * ((df["close"] - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(d_period).mean()

        return {"k": k.iloc[-1] if len(k) > 0 else 50, "d": d.iloc[-1] if len(d) > 0 else 50}

    def calculate_obv(self, df: pd.DataFrame) -> float:
        """計算 OBV"""
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return obv.iloc[-1] if len(obv) > 0 else 0
