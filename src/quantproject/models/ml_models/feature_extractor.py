"""
Feature extraction pipeline integrating LSTM predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

from .lstm_predictor import LSTMPredictor
from .data_preprocessor import TimeSeriesPreprocessor
from .base_model import ModelConfig


logger = logging.getLogger(__name__)


class LSTMFeatureExtractor:
    """
    Extract predictive features using LSTM models
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        prediction_horizons: List[int] = [1, 5, 20],
        feature_prefix: str = 'lstm'
    ):
        """
        Initialize LSTM feature extractor
        
        Args:
            model_path: Path to pre-trained LSTM model
            prediction_horizons: List of prediction horizons
            feature_prefix: Prefix for generated feature names
        """
        self.model_path = model_path
        self.prediction_horizons = prediction_horizons
        self.feature_prefix = feature_prefix
        
        self.models = {}
        self.preprocessors = {}
        self.is_loaded = False
        
    def load_models(self) -> None:
        """Load pre-trained LSTM models"""
        if self.model_path is None:
            raise ValueError("Model path not specified")
        
        model_path = Path(self.model_path)
        
        # Load models for each prediction horizon
        for horizon in self.prediction_horizons:
            horizon_path = model_path / f'horizon_{horizon}'
            
            if horizon_path.exists():
                # Load model
                config = ModelConfig.load(horizon_path / 'config.json')
                model = LSTMPredictor(config)
                model.load(horizon_path)
                
                self.models[horizon] = model
                logger.info(f"Loaded LSTM model for {horizon}-step prediction")
            else:
                logger.warning(f"Model not found for {horizon}-step prediction")
        
        self.is_loaded = True
    
    def train_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'close',
        sequence_length: int = 60,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Train LSTM models for each prediction horizon
        
        Args:
            training_data: Historical data for training
            feature_columns: Features to use for prediction
            target_column: Target variable to predict
            sequence_length: Length of input sequences
            **training_kwargs: Additional training arguments
            
        Returns:
            Training results for each model
        """
        results = {}
        
        for horizon in self.prediction_horizons:
            logger.info(f"Training LSTM for {horizon}-step ahead prediction")
            
            # Create model config
            config = ModelConfig(
                model_type='lstm',
                sequence_length=sequence_length,
                prediction_horizon=horizon,
                feature_columns=feature_columns,
                target_column=target_column,
                hidden_units=[128, 64, 32],
                dropout_rate=0.2,
                batch_size=32,
                epochs=100,
                learning_rate=0.001
            )
            
            # Create model and preprocessor
            model = LSTMPredictor(config)
            preprocessor = TimeSeriesPreprocessor(
                sequence_length=sequence_length,
                prediction_horizon=horizon,
                feature_columns=feature_columns,
                target_column=target_column
            )
            
            # Prepare data
            split_data = preprocessor.create_train_test_split(
                training_data,
                test_size=0.2
            )
            
            # Train model
            from .model_trainer import ModelTrainer
            trainer = ModelTrainer(
                model,
                preprocessor,
                output_dir=f'./models/lstm_horizon_{horizon}'
            )
            
            train_results = trainer.train_model(
                training_data,
                **training_kwargs
            )
            
            # Store model and preprocessor
            self.models[horizon] = model
            self.preprocessors[horizon] = preprocessor
            
            results[f'horizon_{horizon}'] = train_results
        
        self.is_loaded = True
        return results
    
    def extract_features(
        self,
        data: pd.DataFrame,
        include_confidence: bool = True,
        include_trend: bool = True
    ) -> pd.DataFrame:
        """
        Extract LSTM-based features from data
        
        Args:
            data: Input data with required features
            include_confidence: Whether to include prediction confidence
            include_trend: Whether to include trend features
            
        Returns:
            DataFrame with additional LSTM features
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() or train_models() first")
        
        features = data.copy()
        
        for horizon, model in self.models.items():
            preprocessor = self.preprocessors.get(horizon)
            
            if preprocessor is None:
                logger.warning(f"No preprocessor for horizon {horizon}, skipping")
                continue
            
            try:
                # Prepare sequences
                X, _ = preprocessor.transform(data)
                
                if len(X) == 0:
                    continue
                
                # Make predictions
                if include_confidence:
                    predictions, confidence = model.predict(X, return_confidence=True)
                else:
                    predictions = model.predict(X)
                    confidence = None
                
                # Align with original data
                pred_index = data.index[preprocessor.sequence_length:]
                
                # Add prediction features
                feature_name = f'{self.feature_prefix}_pred_{horizon}'
                features.loc[pred_index, feature_name] = predictions
                
                if include_confidence and confidence is not None:
                    features.loc[pred_index, f'{feature_name}_conf_lower'] = confidence[:, 0]
                    features.loc[pred_index, f'{feature_name}_conf_upper'] = confidence[:, 1]
                    features.loc[pred_index, f'{feature_name}_conf_width'] = confidence[:, 1] - confidence[:, 0]
                
                # Add trend features
                if include_trend and len(predictions) > 1:
                    # Prediction slope
                    pred_diff = np.diff(predictions)
                    features.loc[pred_index[1:], f'{feature_name}_slope'] = pred_diff
                    
                    # Prediction acceleration
                    if len(pred_diff) > 1:
                        pred_acc = np.diff(pred_diff)
                        features.loc[pred_index[2:], f'{feature_name}_acceleration'] = pred_acc
                    
                    # Prediction vs current price
                    current_prices = data.loc[pred_index, preprocessor.target_column].values
                    pred_returns = (predictions - current_prices) / current_prices
                    features.loc[pred_index, f'{feature_name}_return'] = pred_returns
                
                logger.info(f"Extracted features for {horizon}-step prediction")
                
            except Exception as e:
                logger.error(f"Error extracting features for horizon {horizon}: {str(e)}")
                continue
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated"""
        feature_names = []
        
        for horizon in self.prediction_horizons:
            base_name = f'{self.feature_prefix}_pred_{horizon}'
            feature_names.append(base_name)
            feature_names.extend([
                f'{base_name}_conf_lower',
                f'{base_name}_conf_upper',
                f'{base_name}_conf_width',
                f'{base_name}_slope',
                f'{base_name}_acceleration',
                f'{base_name}_return'
            ])
        
        return feature_names
    
    def get_live_predictions(
        self,
        historical_data: pd.DataFrame,
        current_features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get live predictions for all horizons
        
        Args:
            historical_data: Recent historical data
            current_features: Current feature values
            
        Returns:
            Dictionary of predictions by horizon
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        
        predictions = {}
        
        for horizon, model in self.models.items():
            preprocessor = self.preprocessors.get(horizon)
            
            if preprocessor is None:
                continue
            
            try:
                # Prepare live data
                X = preprocessor.prepare_live_data(historical_data, current_features)
                
                # Make prediction
                pred = model.predict(X)[0]
                
                # Inverse transform if needed
                if preprocessor.scale_target:
                    pred_original = preprocessor.inverse_transform_predictions(np.array([pred]))[0]
                else:
                    pred_original = pred
                
                # Calculate expected return
                current_price = historical_data[preprocessor.target_column].iloc[-1]
                expected_return = (pred_original - current_price) / current_price
                
                predictions[f'horizon_{horizon}'] = {
                    'prediction': float(pred_original),
                    'current_price': float(current_price),
                    'expected_return': float(expected_return),
                    'horizon_days': horizon
                }
                
            except Exception as e:
                logger.error(f"Error getting prediction for horizon {horizon}: {str(e)}")
                continue
        
        return predictions


class MultiModelFeatureExtractor:
    """
    Combine features from multiple models
    """
    
    def __init__(self):
        """Initialize multi-model feature extractor"""
        self.extractors = {}
        
    def add_extractor(self, name: str, extractor: Any) -> None:
        """Add a feature extractor"""
        self.extractors[name] = extractor
    
    def extract_all_features(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from all registered extractors
        
        Args:
            data: Input data
            **kwargs: Arguments passed to extractors
            
        Returns:
            DataFrame with all extracted features
        """
        features = data.copy()
        
        for name, extractor in self.extractors.items():
            logger.info(f"Extracting features from {name}")
            
            try:
                if hasattr(extractor, 'extract_features'):
                    extractor_features = extractor.extract_features(data, **kwargs)
                    
                    # Add new features
                    new_cols = [col for col in extractor_features.columns if col not in features.columns]
                    features = pd.concat([features, extractor_features[new_cols]], axis=1)
                    
                else:
                    logger.warning(f"Extractor {name} does not have extract_features method")
                    
            except Exception as e:
                logger.error(f"Error in {name} extractor: {str(e)}")
                continue
        
        return features
    
    def get_all_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names from all extractors"""
        feature_names = {}
        
        for name, extractor in self.extractors.items():
            if hasattr(extractor, 'get_feature_names'):
                feature_names[name] = extractor.get_feature_names()
            else:
                feature_names[name] = []
        
        return feature_names