"""
Feature Engineering Module for Financial Data
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

from .indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create and transform features for machine learning models
    """

    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize feature engineer

        Args:
            scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler()
        self.feature_columns = []

    def _get_scaler(self):
        """Get appropriate scaler based on scaling method"""
        if self.scaling_method == "standard":
            return StandardScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        elif self.scaling_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional price features
        """
        result = df.copy()

        # Price ratios
        result["high_low_ratio"] = result["high"] / result["low"]
        result["close_open_ratio"] = result["close"] / result["open"]

        # Price ranges
        result["daily_range"] = result["high"] - result["low"]
        result["daily_range_pct"] = result["daily_range"] / result["close"]

        # Candlestick patterns
        result["body_size"] = (result["close"] - result["open"]).abs()
        result["upper_shadow"] = result["high"] - result[["open", "close"]].max(axis=1)
        result["lower_shadow"] = result[["open", "close"]].min(axis=1) - result["low"]

        # Price position within range
        result["close_position"] = (result["close"] - result["low"]) / (
            result["high"] - result["low"]
        )

        # Gap features
        result["gap"] = result["open"] - result["close"].shift(1)
        result["gap_pct"] = result["gap"] / result["close"].shift(1)

        return result

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with additional volume features
        """
        result = df.copy()

        if "volume" not in result.columns:
            logger.warning("No volume column found, skipping volume features")
            return result

        # Volume moving averages
        result["volume_sma_10"] = result["volume"].rolling(window=10).mean()
        result["volume_sma_20"] = result["volume"].rolling(window=20).mean()

        # Volume ratios
        result["volume_ratio_10"] = result["volume"] / result["volume_sma_10"]
        result["volume_ratio_20"] = result["volume"] / result["volume_sma_20"]

        # Price-volume features
        if "close" in result.columns:
            result["price_volume"] = result["close"] * result["volume"]
            result["price_volume_sma"] = result["price_volume"].rolling(window=10).mean()

        # Volume change
        result["volume_change"] = result["volume"].pct_change()

        return result

    def create_return_features(
        self, df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create return-based features

        Args:
            df: DataFrame with price data
            periods: List of periods for return calculation

        Returns:
            DataFrame with return features
        """
        result = df.copy()

        if "close" not in result.columns:
            logger.warning("No close column found, skipping return features")
            return result

        # Simple returns
        for period in periods:
            result[f"return_{period}"] = result["close"].pct_change(period)

        # Log returns
        for period in periods:
            result[f"log_return_{period}"] = np.log(result["close"] / result["close"].shift(period))

        # Cumulative returns
        result["cumulative_return"] = (1 + result["return_1"]).cumprod() - 1

        # Rolling statistics
        for period in [10, 20]:
            result[f"return_mean_{period}"] = result["return_1"].rolling(window=period).mean()
            result[f"return_std_{period}"] = result["return_1"].rolling(window=period).std()
            result[f"return_skew_{period}"] = result["return_1"].rolling(window=period).skew()
            result[f"return_kurt_{period}"] = result["return_1"].rolling(window=period).kurt()

        return result

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with volatility features
        """
        result = df.copy()

        # Historical volatility (different windows)
        for period in [5, 10, 20]:
            if "return_1" in result.columns:
                result[f"volatility_{period}"] = result["return_1"].rolling(window=period).std()
            else:
                returns = result["close"].pct_change()
                result[f"volatility_{period}"] = returns.rolling(window=period).std()

        # Parkinson volatility (using high-low)
        if all(col in result.columns for col in ["high", "low"]):
            result["parkinson_vol"] = (
                np.sqrt((1 / (4 * np.log(2))) * np.power(np.log(result["high"] / result["low"]), 2))
                .rolling(window=20)
                .mean()
            )

        # Garman-Klass volatility
        if all(col in result.columns for col in ["high", "low", "close", "open"]):
            result["garman_klass_vol"] = (
                np.sqrt(
                    0.5 * np.power(np.log(result["high"] / result["low"]), 2)
                    - (2 * np.log(2) - 1) * np.power(np.log(result["close"] / result["open"]), 2)
                )
                .rolling(window=20)
                .mean()
            )

        return result

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame with time features
        """
        result = df.copy()

        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("Index is not datetime, skipping time features")
            return result

        # Time of day features
        result["hour"] = result.index.hour
        result["minute"] = result.index.minute
        result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

        # Day of week features
        result["dayofweek"] = result.index.dayofweek
        result["is_monday"] = (result["dayofweek"] == 0).astype(int)
        result["is_friday"] = (result["dayofweek"] == 4).astype(int)

        # Month features
        result["month"] = result.index.month
        result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
        result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

        # Quarter features
        result["quarter"] = result.index.quarter

        # Trading session features (assuming US market hours)
        result["is_market_open"] = ((result["hour"] >= 9) & (result["hour"] < 16)).astype(int)

        result["is_pre_market"] = ((result["hour"] >= 4) & (result["hour"] < 9)).astype(int)

        result["is_after_hours"] = ((result["hour"] >= 16) & (result["hour"] < 20)).astype(int)

        return result

    def create_lag_features(
        self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lagged features

        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            for lag in lags:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)

        return result

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
        functions: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """
        Create rolling window features

        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            functions: List of aggregation functions

        Returns:
            DataFrame with rolling features
        """
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            for window in windows:
                for func in functions:
                    feature_name = f"{col}_rolling_{window}_{func}"

                    if func == "mean":
                        result[feature_name] = result[col].rolling(window=window).mean()
                    elif func == "std":
                        result[feature_name] = result[col].rolling(window=window).std()
                    elif func == "min":
                        result[feature_name] = result[col].rolling(window=window).min()
                    elif func == "max":
                        result[feature_name] = result[col].rolling(window=window).max()

        return result

    def engineer_all_features(
        self, df: pd.DataFrame, feature_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create all feature groups

        Args:
            df: Input DataFrame with OHLCV data
            feature_groups: List of feature groups to create (None for all)

        Returns:
            DataFrame with all engineered features
        """
        result = df.copy()

        # Default to all feature groups
        if feature_groups is None:
            feature_groups = ["price", "volume", "returns", "volatility", "time", "technical"]

        # Create features based on groups
        if "price" in feature_groups:
            result = self.create_price_features(result)

        if "volume" in feature_groups:
            result = self.create_volume_features(result)

        if "returns" in feature_groups:
            result = self.create_return_features(result)

        if "volatility" in feature_groups:
            result = self.create_volatility_features(result)

        if "time" in feature_groups:
            result = self.create_time_features(result)

        if "technical" in feature_groups:
            # Add technical indicators
            result = TechnicalIndicators.calculate_all_indicators(result)

        # Store feature columns (excluding original OHLCV)
        original_cols = ["open", "high", "low", "close", "volume"]
        self.feature_columns = [col for col in result.columns if col not in original_cols]

        return result

    def scale_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using the configured scaler

        Args:
            df: Input DataFrame
            columns: Columns to scale (None for all numeric)
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            DataFrame with scaled features
        """
        result = df.copy()

        # Determine columns to scale
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()

        # Remove any columns with all NaN
        columns = [col for col in columns if not result[col].isna().all()]

        if not columns:
            logger.warning("No columns to scale")
            return result

        # Scale features
        if fit:
            scaled_data = self.scaler.fit_transform(result[columns])
        else:
            scaled_data = self.scaler.transform(result[columns])

        # Update DataFrame
        result[columns] = scaled_data

        return result
