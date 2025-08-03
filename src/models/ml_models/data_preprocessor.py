"""
Data preprocessing for time series models
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging


logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocessor for time series data"""
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'close',
        scaling_method: str = 'standard',
        scale_target: bool = True
    ):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict
            feature_columns: Columns to use as features
            target_column: Column to predict
            scaling_method: Method for scaling ('standard', 'minmax', 'robust')
            scale_target: Whether to scale target values
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaling_method = scaling_method
        self.scale_target = scale_target
        
        # Scalers
        self.feature_scaler = self._get_scaler(scaling_method)
        self.target_scaler = self._get_scaler(scaling_method) if scale_target else None
        
        # State
        self.is_fitted = False
        self._feature_indices = None
    
    def _get_scaler(self, method: str):
        """Get scaler based on method"""
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit preprocessor on training data
        
        Args:
            data: Training data
        """
        # Determine feature columns if not specified
        if self.feature_columns is None:
            # Use all numeric columns except target
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_cols if col != self.target_column]
        
        # Validate columns exist
        missing_cols = set(self.feature_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Fit scalers
        feature_data = data[self.feature_columns].values
        self.feature_scaler.fit(feature_data)
        
        if self.scale_target:
            target_data = data[self.target_column].values.reshape(-1, 1)
            self.target_scaler.fit(target_data)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted with {len(self.feature_columns)} features")
    
    def transform(
        self,
        data: pd.DataFrame,
        return_pandas: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]]:
        """
        Transform data into sequences
        
        Args:
            data: Data to transform
            return_pandas: Whether to return pandas objects
            
        Returns:
            Tuple of (X_sequences, y_values)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale features
        feature_data = data[self.feature_columns].values
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Scale target if needed
        if self.target_column in data.columns:
            target_data = data[self.target_column].values
            if self.scale_target:
                target_data = self.target_scaler.transform(target_data.reshape(-1, 1)).flatten()
        else:
            target_data = None
        
        # Create sequences
        X_sequences = []
        y_values = []
        
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            # Input sequence
            X_sequences.append(scaled_features[i - self.sequence_length:i])
            
            # Target value
            if target_data is not None:
                y_values.append(target_data[i + self.prediction_horizon - 1])
        
        X_sequences = np.array(X_sequences)
        y_values = np.array(y_values) if y_values else None
        
        if return_pandas:
            # Create DataFrame with appropriate index
            sequence_index = data.index[self.sequence_length:len(data) - self.prediction_horizon + 1]
            
            # Flatten sequences for DataFrame
            X_df = pd.DataFrame(
                X_sequences.reshape(len(X_sequences), -1),
                index=sequence_index
            )
            
            y_series = pd.Series(y_values, index=sequence_index) if y_values is not None else None
            
            return X_df, y_series
        
        return X_sequences, y_values
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        if not self.scale_target or self.target_scaler is None:
            return predictions
        
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(predictions).flatten()
    
    def prepare_live_data(
        self,
        historical_data: pd.DataFrame,
        current_features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Prepare data for live prediction
        
        Args:
            historical_data: Historical data (at least sequence_length rows)
            current_features: Optional current feature values
            
        Returns:
            Prepared sequence for prediction
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before use")
        
        if len(historical_data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows of historical data, "
                f"got {len(historical_data)}"
            )
        
        # Take last sequence_length rows
        data = historical_data.tail(self.sequence_length).copy()
        
        # Update with current features if provided
        if current_features:
            for col, value in current_features.items():
                if col in self.feature_columns:
                    data.iloc[-1][col] = value
        
        # Scale features
        feature_data = data[self.feature_columns].values
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Return as single sequence
        return scaled_features.reshape(1, self.sequence_length, len(self.feature_columns))
    
    def create_train_test_split(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        gap: int = 0
    ) -> Dict[str, Any]:
        """
        Create train/test split for time series
        
        Args:
            data: Full dataset
            test_size: Proportion of data for testing
            gap: Gap between train and test sets
            
        Returns:
            Dictionary with train/test data
        """
        # Calculate split point
        split_point = int(len(data) * (1 - test_size)) - gap
        
        # Split data
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point + gap:]
        
        # Fit on training data
        self.fit(train_data)
        
        # Transform both sets
        X_train, y_train = self.transform(train_data)
        X_test, y_test = self.transform(test_data)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_index': train_data.index[self.sequence_length:],
            'test_index': test_data.index[self.sequence_length:],
            'feature_names': self.feature_columns,
            'split_point': split_point
        }
    
    def add_technical_features(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add technical indicators as features
        
        Args:
            data: OHLCV data
            indicators: List of indicators to add
            
        Returns:
            Data with additional features
        """
        if indicators is None:
            indicators = ['returns', 'log_returns', 'volatility', 'volume_ratio']
        
        df = data.copy()
        
        # Price returns
        if 'returns' in indicators:
            df['returns'] = df['close'].pct_change()
            df['returns_lag1'] = df['returns'].shift(1)
            df['returns_lag2'] = df['returns'].shift(2)
        
        # Log returns
        if 'log_returns' in indicators:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        if 'volatility' in indicators:
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_60'] = df['returns'].rolling(window=60).std()
        
        # Volume features
        if 'volume_ratio' in indicators and 'volume' in df.columns:
            df['volume_sma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
        
        # Price ratios
        if 'price_ratios' in indicators:
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        if 'moving_averages' in indicators:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['price_to_sma20'] = df['close'] / df['sma_20']
            df['price_to_sma50'] = df['close'] / df['sma_50']
        
        # Drop NaN rows created by indicators
        df = df.dropna()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns.copy()
    
    def save_config(self, filepath: str) -> None:
        """Save preprocessor configuration"""
        import json
        
        config = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'scaling_method': self.scaling_method,
            'scale_target': self.scale_target,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'TimeSeriesPreprocessor':
        """Load preprocessor configuration"""
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        is_fitted = config.pop('is_fitted', False)
        preprocessor = cls(**config)
        
        if is_fitted:
            logger.warning(
                "Loaded preprocessor was fitted, but scalers not restored. "
                "Call fit() again with training data."
            )
        
        return preprocessor