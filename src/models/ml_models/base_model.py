"""
Base model classes for machine learning predictors
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
import json
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    
    # Model architecture
    model_type: str = "lstm"
    input_features: int = 10
    sequence_length: int = 60
    hidden_units: List[int] = None
    dropout_rate: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Feature configuration
    feature_columns: List[str] = None
    target_column: str = "close"
    prediction_horizon: int = 1
    
    # Preprocessing
    scale_features: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust
    
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [128, 64, 32]
        if self.feature_columns is None:
            self.feature_columns = ["close", "volume", "rsi", "macd"]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        config_dict = {
            k: v for k, v in self.__dict__.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelConfig':
        """Load configuration from JSON file"""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


class BasePredictor(ABC):
    """Abstract base class for financial predictors"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize predictor
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_importance = {}
        self.scaler = None
        
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training arguments
            
        Returns:
            Training history and metrics
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Input features
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Predictions and optionally confidence intervals
        """
        pass
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions))
            true_direction = np.sign(np.diff(y_test))
            directional_accuracy = np.mean(pred_direction == true_direction)
        else:
            directional_accuracy = 0.0
        
        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> None:
        """Load model from file"""
        pass
    
    def predict_next(
        self,
        historical_data: pd.DataFrame,
        steps: int = 1
    ) -> pd.DataFrame:
        """
        Predict next values given historical data
        
        Args:
            historical_data: Historical price/feature data
            steps: Number of steps to predict ahead
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # This is a convenience method that should be implemented
        # by specific predictors based on their requirements
        raise NotImplementedError("Subclasses should implement predict_next")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process"""
        return {
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }


class EnsemblePredictor(BasePredictor):
    """Ensemble of multiple predictors"""
    
    def __init__(
        self,
        predictors: List[BasePredictor],
        weights: Optional[List[float]] = None,
        voting: str = 'average'
    ):
        """
        Initialize ensemble predictor
        
        Args:
            predictors: List of base predictors
            weights: Weights for each predictor
            voting: Voting method ('average', 'weighted', 'median')
        """
        # Create a combined config
        config = ModelConfig(model_type='ensemble')
        super().__init__(config)
        
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)
        self.voting = voting
        
        if len(self.weights) != len(self.predictors):
            raise ValueError("Number of weights must match number of predictors")
    
    def build_model(self) -> None:
        """Build all sub-models"""
        for predictor in self.predictors:
            predictor.build_model()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train all predictors"""
        all_histories = {}
        
        for i, predictor in enumerate(self.predictors):
            logger.info(f"Training predictor {i+1}/{len(self.predictors)}: {predictor.config.model_type}")
            
            history = predictor.train(X_train, y_train, X_val, y_val, **kwargs)
            all_histories[f'predictor_{i}'] = history
        
        self.is_trained = True
        self.training_history = all_histories
        
        return all_histories
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Collect predictions from all models
        all_predictions = []
        
        for predictor in self.predictors:
            pred = predictor.predict(X, return_confidence=False)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Combine predictions based on voting method
        if self.voting == 'average':
            predictions = np.mean(all_predictions, axis=0)
        elif self.voting == 'weighted':
            predictions = np.average(all_predictions, axis=0, weights=self.weights)
        elif self.voting == 'median':
            predictions = np.median(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown voting method: {self.voting}")
        
        if return_confidence:
            # Use std of predictions as confidence
            confidence = np.std(all_predictions, axis=0)
            return predictions, confidence
        
        return predictions
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save ensemble model"""
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble config
        ensemble_config = {
            'voting': self.voting,
            'weights': self.weights,
            'num_predictors': len(self.predictors)
        }
        
        with open(filepath / 'ensemble_config.json', 'w') as f:
            json.dump(ensemble_config, f)
        
        # Save each predictor
        for i, predictor in enumerate(self.predictors):
            predictor.save(filepath / f'predictor_{i}')
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load ensemble model"""
        # This would need to be implemented based on saved structure
        raise NotImplementedError("Load method for ensemble not implemented")