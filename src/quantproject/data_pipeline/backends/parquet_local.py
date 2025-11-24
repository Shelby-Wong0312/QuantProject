"""
Local Parquet Backend for Historical Data
Reads from scripts/download/historical_data/daily/*.parquet
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import IDataBackend


class ParquetLocalBackend(IDataBackend):
    """Backend for loading OHLCV data from local parquet files."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize ParquetLocalBackend

        Args:
            data_dir: Directory containing parquet files.
                     Defaults to scripts/download/historical_data/daily/
        """
        if data_dir is None:
            # Default to project root / scripts/download/historical_data/daily
            current_file = Path(__file__)
            project_root = current_file.parents[5]  # Go up to project root
            data_dir = project_root / "scripts" / "download" / "historical_data" / "daily"

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

    def get_bars(
        self, symbol: str, start: str, end: str, timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Load OHLCV bars from local parquet file

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start: Start date (e.g., '2015-01-01')
            end: End date (e.g., '2022-12-31')
            timeframe: Timeframe (only 'daily'/'1d' supported for parquet data)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Empty DataFrame if symbol not found or error occurs
        """
        # Normalize timeframe
        tf_lower = timeframe.strip().lower()
        if tf_lower not in {"1d", "daily", "day"}:
            # Parquet files only contain daily data
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Construct file path: symbol_daily.parquet or SYMBOL_daily.parquet
        file_path = None
        for candidate in [
            self.data_dir / f"{symbol}_daily.parquet",
            self.data_dir / f"{symbol.upper()}_daily.parquet",
            self.data_dir / f"{symbol.lower()}_daily.parquet",
        ]:
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            # Read parquet file
            df = pd.read_parquet(file_path)

            # Handle timestamp column
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()

            # Filter date range
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Select and rename columns
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame(columns=required_cols)

            result = df[required_cols].copy()

            # Ensure UTC timezone for consistency
            if result.index.tz is None:
                result.index = result.index.tz_localize("UTC")
            else:
                result.index = result.index.tz_convert("UTC")

            return result

        except Exception as e:
            print(f"Error loading {symbol} from parquet: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


__all__ = ["ParquetLocalBackend"]
