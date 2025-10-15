"""
Data Cleaning Pipeline for Financial Data
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and preprocess financial market data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data cleaner

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.missing_threshold = self.config.get("missing_threshold", 0.1)  # 10% max missing
        self.outlier_std = self.config.get("outlier_std", 5)  # 5 standard deviations

    def clean_ohlcv_data(
        self,
        df: pd.DataFrame,
        remove_weekends: bool = True,
        remove_holidays: bool = False,
        forward_fill_limit: int = 5,
    ) -> pd.DataFrame:
        """
        Clean OHLCV (Open, High, Low, Close, Volume) data

        Args:
            df: DataFrame with OHLCV columns and datetime index
            remove_weekends: Whether to remove weekend data
            remove_holidays: Whether to remove market holidays
            forward_fill_limit: Maximum consecutive forward fills

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        original_len = len(df)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df.set_index("datetime", inplace=True)
            elif "date" in df.columns:
                df.set_index("date", inplace=True)
            else:
                raise ValueError("DataFrame must have datetime index or datetime/date column")

        # Sort by datetime
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Basic column validation
        required_cols = ["open", "high", "low", "close", "volume"]
        df.columns = df.columns.str.lower()

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")

        # Remove weekends if requested
        if remove_weekends and pd.api.types.is_datetime64_any_dtype(df.index):
            df = df[df.index.dayofweek < 5]

        # Handle missing values
        df = self._handle_missing_values(df, forward_fill_limit)

        # Remove invalid OHLC relationships
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            invalid_mask = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )

            if invalid_mask.any():
                logger.warning(
                    f"Removing {invalid_mask.sum()} rows with invalid OHLC relationships"
                )
                df = df[~invalid_mask]

        # Handle negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                if negative_mask.any():
                    logger.warning(
                        f"Found {negative_mask.sum()} negative values in {col}, setting to NaN"
                    )
                    df.loc[negative_mask, col] = np.nan

        # Handle negative volume
        if "volume" in df.columns:
            negative_volume = df["volume"] < 0
            if negative_volume.any():
                logger.warning(
                    f"Found {negative_volume.sum()} negative volume values, setting to 0"
                )
                df.loc[negative_volume, "volume"] = 0

        # Remove outliers
        df = self._remove_outliers(df)

        # Final forward fill for any remaining NaNs
        df.fillna(method="ffill", limit=forward_fill_limit, inplace=True)

        # Drop any remaining rows with NaN in critical columns
        critical_cols = ["open", "high", "low", "close"]
        existing_critical = [col for col in critical_cols if col in df.columns]
        if existing_critical:
            df.dropna(subset=existing_critical, inplace=True)

        logger.info(f"Data cleaning complete. Rows: {original_len} -> {len(df)}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame, forward_fill_limit: int = 5) -> pd.DataFrame:
        """
        Handle missing values in the dataframe

        Args:
            df: Input DataFrame
            forward_fill_limit: Maximum consecutive forward fills

        Returns:
            DataFrame with handled missing values
        """
        # Check missing value percentages
        missing_pct = df.isnull().sum() / len(df)

        # Drop columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        if cols_to_drop:
            logger.warning(
                f"Dropping columns with >{self.missing_threshold*100}% missing: {cols_to_drop}"
            )
            df.drop(columns=cols_to_drop, inplace=True)

        # Forward fill for time series continuity
        df.fillna(method="ffill", limit=forward_fill_limit, inplace=True)

        # Backward fill for any remaining at the start
        df.fillna(method="bfill", limit=1, inplace=True)

        return df

    def _remove_outliers(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """
        Remove outliers from price data

        Args:
            df: Input DataFrame
            method: Outlier detection method ('zscore' or 'iqr')

        Returns:
            DataFrame with outliers removed
        """
        price_cols = ["open", "high", "low", "close"]
        existing_cols = [col for col in price_cols if col in df.columns]

        if not existing_cols:
            return df

        if method == "zscore":
            # Calculate returns to detect price spikes
            for col in existing_cols:
                if col in df.columns:
                    returns = df[col].pct_change()
                    z_scores = np.abs((returns - returns.mean()) / returns.std())
                    outlier_mask = z_scores > self.outlier_std

                    if outlier_mask.any():
                        logger.info(
                            f"Found {outlier_mask.sum()} outliers in {col} using z-score method"
                        )
                        df.loc[outlier_mask, col] = np.nan

        elif method == "iqr":
            # Interquartile range method
            for col in existing_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

                    if outlier_mask.any():
                        logger.info(
                            f"Found {outlier_mask.sum()} outliers in {col} using IQR method"
                        )
                        df.loc[outlier_mask, col] = np.nan

        return df

    def resample_data(
        self,
        df: pd.DataFrame,
        target_frequency: str = "1H",
        aggregation_rules: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Resample time series data to target frequency

        Args:
            df: Input DataFrame with datetime index
            target_frequency: Target frequency (e.g., '1H', '1D', '5T')
            aggregation_rules: Custom aggregation rules for each column

        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for resampling")

        # Default aggregation rules
        if aggregation_rules is None:
            aggregation_rules = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

        # Apply aggregation rules for existing columns
        agg_dict = {col: rule for col, rule in aggregation_rules.items() if col in df.columns}

        # Resample
        resampled = df.resample(target_frequency).agg(agg_dict)

        # Remove any all-NaN rows
        resampled.dropna(how="all", inplace=True)

        return resampled

    def align_multiple_datasets(
        self, datasets: Dict[str, pd.DataFrame], method: str = "inner"
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple datasets to common datetime index

        Args:
            datasets: Dictionary of dataset name to DataFrame
            method: Alignment method ('inner', 'outer', 'forward_fill')

        Returns:
            Dictionary of aligned DataFrames
        """
        if not datasets:
            return {}

        # Get all unique timestamps
        all_timestamps = set()
        for df in datasets.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.update(df.index)

        if not all_timestamps:
            logger.warning("No datetime indices found in datasets")
            return datasets

        # Create common index
        common_index = pd.DatetimeIndex(sorted(all_timestamps))

        aligned_datasets = {}

        for name, df in datasets.items():
            if method == "inner":
                # Keep only common timestamps
                aligned_df = df.reindex(df.index.intersection(common_index))
            elif method == "outer":
                # Include all timestamps, forward fill missing
                aligned_df = df.reindex(common_index, method="ffill")
            elif method == "forward_fill":
                # Include all timestamps, forward fill with limit
                aligned_df = df.reindex(common_index, method="ffill", limit=5)
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            aligned_datasets[name] = aligned_df

        return aligned_datasets
