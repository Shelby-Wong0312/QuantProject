"""
Complete Data Processing Pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

from .data_cleaner import DataCleaner
from .data_validator import DataValidator
from .feature_engineering import FeatureEngineer
from .indicators import TechnicalIndicators


logger = logging.getLogger(__name__)


class DataProcessingPipeline:
    """
    End-to-end data processing pipeline for financial data
    """

    def __init__(
        self,
        cleaning_config: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        scaling_method: str = "standard",
    ):
        """
        Initialize data processing pipeline

        Args:
            cleaning_config: Configuration for data cleaning
            validation_config: Configuration for data validation
            scaling_method: Method for feature scaling
        """
        self.cleaner = DataCleaner(cleaning_config)
        self.validator = DataValidator(validation_config)
        self.feature_engineer = FeatureEngineer(scaling_method)
        self.processed_data = None
        self.validation_report = None

    def process(
        self,
        data: Union[pd.DataFrame, str, Path],
        validate_first: bool = True,
        clean_data: bool = True,
        engineer_features: bool = True,
        calculate_indicators: bool = True,
        scale_features: bool = True,
        feature_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process data through the complete pipeline

        Args:
            data: Input data (DataFrame or path to CSV file)
            validate_first: Whether to validate data before processing
            clean_data: Whether to clean the data
            engineer_features: Whether to engineer features
            calculate_indicators: Whether to calculate technical indicators
            scale_features: Whether to scale features
            feature_groups: Specific feature groups to engineer

        Returns:
            Processed DataFrame
        """
        # Load data if path provided
        if isinstance(data, (str, Path)):
            logger.info(f"Loading data from {data}")
            df = pd.read_csv(data, parse_dates=True, index_col=0)
        else:
            df = data.copy()

        # Initial validation
        if validate_first:
            logger.info("Validating input data...")
            is_valid, report = self.validator.validate_ohlcv(df)
            self.validation_report = report

            if not is_valid:
                logger.warning(
                    "Data validation failed. Check validation_report for details."
                )
                # Continue with processing but log the issues
                for issue in report["issues"]:
                    logger.error(f"Validation issue: {issue}")

        # Clean data
        if clean_data:
            logger.info("Cleaning data...")
            df = self.cleaner.clean_ohlcv_data(df)

        # Calculate technical indicators
        if calculate_indicators:
            logger.info("Calculating technical indicators...")
            df = TechnicalIndicators.calculate_all_indicators(df)

        # Engineer features
        if engineer_features:
            logger.info("Engineering features...")
            df = self.feature_engineer.engineer_all_features(df, feature_groups)

        # Scale features
        if scale_features:
            logger.info("Scaling features...")
            # Don't scale OHLCV columns
            feature_cols = [
                col
                for col in df.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            df[feature_cols] = self.feature_engineer.scale_features(
                df[feature_cols], fit=True
            )

        # Final validation
        if validate_first:
            logger.info("Performing final validation...")
            is_valid_final, report_final = self.validator.validate_ohlcv(
                df, check_gaps=False
            )

            logger.info(
                f"Final data quality score: {report_final['quality_score']:.2f}/100"
            )

        self.processed_data = df
        return df

    def process_for_training(
        self,
        data: Union[pd.DataFrame, str, Path],
        target_column: str = "close",
        prediction_horizon: int = 1,
        train_size: float = 0.8,
        create_sequences: bool = True,
        sequence_length: int = 60,
    ) -> Dict[str, np.ndarray]:
        """
        Process data specifically for model training

        Args:
            data: Input data
            target_column: Column to predict
            prediction_horizon: How many periods ahead to predict
            train_size: Proportion of data for training
            create_sequences: Whether to create sequences for LSTM/RNN
            sequence_length: Length of sequences

        Returns:
            Dictionary with training arrays
        """
        # Process data through pipeline
        df = self.process(data)

        # Create target variable (future return)
        df[f"target_return_{prediction_horizon}"] = (
            df[target_column].shift(-prediction_horizon) / df[target_column] - 1
        )

        # Drop rows with NaN target
        df = df.dropna(subset=[f"target_return_{prediction_horizon}"])

        # Separate features and target
        feature_cols = [col for col in df.columns if not col.startswith("target_")]
        X = df[feature_cols].values
        y = df[f"target_return_{prediction_horizon}"].values

        # Create train/test split
        split_idx = int(len(df) * train_size)

        result = {
            "X_train": X[:split_idx],
            "X_test": X[split_idx:],
            "y_train": y[:split_idx],
            "y_test": y[split_idx:],
            "feature_names": feature_cols,
            "index_train": df.index[:split_idx],
            "index_test": df.index[split_idx:],
        }

        # Create sequences if requested
        if create_sequences:
            X_train_seq, y_train_seq = self._create_sequences(
                result["X_train"], result["y_train"], sequence_length
            )
            X_test_seq, y_test_seq = self._create_sequences(
                result["X_test"], result["y_test"], sequence_length
            )

            result["X_train_sequences"] = X_train_seq
            result["y_train_sequences"] = y_train_seq
            result["X_test_sequences"] = X_test_seq
            result["y_test_sequences"] = y_test_seq

        return result

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models

        Args:
            X: Feature array
            y: Target array
            sequence_length: Length of each sequence

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []

        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i - sequence_length : i])
            y_sequences.append(y[i])

        return np.array(X_sequences), np.array(y_sequences)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature names and basic statistics

        Returns:
            DataFrame with feature information
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process() first.")

        feature_info = []

        for col in self.processed_data.columns:
            info = {
                "feature": col,
                "dtype": str(self.processed_data[col].dtype),
                "missing_pct": (
                    self.processed_data[col].isna().sum() / len(self.processed_data)
                )
                * 100,
                "unique_values": self.processed_data[col].nunique(),
                "mean": (
                    self.processed_data[col].mean()
                    if np.issubdtype(self.processed_data[col].dtype, np.number)
                    else None
                ),
                "std": (
                    self.processed_data[col].std()
                    if np.issubdtype(self.processed_data[col].dtype, np.number)
                    else None
                ),
            }
            feature_info.append(info)

        return pd.DataFrame(feature_info)

    def save_processed_data(self, filepath: Union[str, Path]) -> None:
        """
        Save processed data to file

        Args:
            filepath: Path to save the data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process() first.")

        self.processed_data.to_csv(filepath)
        logger.info(f"Processed data saved to {filepath}")

    def get_validation_summary(self) -> str:
        """
        Get validation summary

        Returns:
            Validation summary string
        """
        if self.validation_report is None:
            return "No validation report available. Run process() with validate_first=True."

        return self.validator.generate_validation_report()
