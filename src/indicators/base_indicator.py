"""
Base Indicator Class - Abstract base class for all technical indicators
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators"""

    def __init__(self, period: int = 14, use_cache: bool = True):
        """
        Initialize base indicator

        Args:
            period: Default period for calculations
            use_cache: Whether to use caching for calculations
        """
        self.period = period
        self.use_cache = use_cache
        self._cache = {}
        self.name = self.__class__.__name__

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator values

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with indicator values
        """
        pass

    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicator

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signal columns (buy, sell, hold)
        """
        pass

    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate input data

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        if not isinstance(data, pd.DataFrame):
            logger.error(f"{self.name}: Input must be a pandas DataFrame")
            return False

        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"{self.name}: Missing required columns: {missing_cols}")
            return False

        if len(data) < self.period:
            logger.error(f"{self.name}: Insufficient data. Need at least {self.period} periods")
            return False

        if data.isnull().any().any():
            logger.warning(f"{self.name}: Data contains null values")

        return True

    def _get_cache_key(self, data: pd.DataFrame, **kwargs) -> str:
        """Generate cache key for data and parameters"""
        if not self.use_cache:
            return None

        # Create hash from data and parameters
        data_str = f"{data.index[0]}_{data.index[-1]}_{len(data)}"
        params_str = f"{self.period}_{kwargs}"
        key = hashlib.md5(f"{data_str}_{params_str}".encode()).hexdigest()
        return key

    def _get_from_cache(self, key: str) -> Optional[pd.Series]:
        """Get cached result if exists"""
        if key and key in self._cache:
            logger.debug(f"{self.name}: Using cached result")
            return self._cache[key].copy()
        return None

    def _save_to_cache(self, key: str, result: pd.Series):
        """Save result to cache"""
        if key and self.use_cache:
            self._cache[key] = result.copy()

    def batch_calculate(self, stocks_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Calculate indicator for multiple stocks

        Args:
            stocks_data: Dictionary with symbol as key and DataFrame as value

        Returns:
            Dictionary with symbol as key and indicator Series as value
        """
        results = {}

        for symbol, data in stocks_data.items():
            try:
                if self.validate(data):
                    results[symbol] = self.calculate(data)
                else:
                    logger.warning(f"Skipping {symbol} due to validation failure")
            except Exception as e:
                logger.error(f"Error calculating {self.name} for {symbol}: {e}")

        return results

    def plot(self, data: pd.DataFrame, indicator_values: pd.Series = None):
        """
        Plot the indicator (to be implemented by specific indicators)

        Args:
            data: OHLCV data
            indicator_values: Pre-calculated indicator values (optional)
        """
        import matplotlib.pyplot as plt

        if indicator_values is None:
            indicator_values = self.calculate(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot price
        ax1.plot(data.index, data["close"], label="Close Price", color="black")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot indicator
        ax2.plot(data.index, indicator_values, label=self.name, color="blue")
        ax2.set_ylabel(self.name)
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"{self.name} Indicator Analysis")
        plt.tight_layout()
        return fig

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {"name": self.name, "period": self.period, "use_cache": self.use_cache}

    def describe(self) -> str:
        """Get indicator description"""
        return f"{self.name} indicator with period {self.period}"
