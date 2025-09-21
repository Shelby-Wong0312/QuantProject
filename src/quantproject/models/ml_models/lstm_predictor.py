"""
LSTM-based trend prediction model
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l1_l2

from .base_model import BasePredictor, ModelConfig


logger = logging.getLogger(__name__)


class LSTMPredictor(BasePredictor):
    """LSTM model for financial time series prediction"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LSTM predictor
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model = None
        self.training_history = None
        
    def build_model(self) -> None:
        """Build LSTM model architecture"""
        # Input shape: (batch_size, sequence_length, n_features)
        inputs = keras.Input(
            shape=(self.config.sequence_length, self.config.input_features)
        )
        
        x = inputs
        
        # Build LSTM layers
        for i, units in enumerate(self.config.hidden_units[:-1]):
            x = layers.LSTM(
                units=units,
                return_sequences=True,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                name=f'lstm_{i}'
            )(x)
            x = layers.BatchNormalization()(x)
        
        # Last LSTM layer (no return_sequences)
        x = layers.LSTM(
            units=self.config.hidden_units[-1],
            return_sequences=False,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='lstm_final'
        )(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Additional dense layers for feature extraction
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_predictor')
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"Built LSTM model with {self.model.count_params():,} parameters")
        
    def build_attention_model(self) -> None:
        """Build LSTM model with attention mechanism"""
        inputs = keras.Input(
            shape=(self.config.sequence_length, self.config.input_features)
        )
        
        # LSTM layers
        lstm_out, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
            layers.LSTM(
                units=self.config.hidden_units[0],
                return_sequences=True,
                return_state=True,
                dropout=self.config.dropout_rate
            ),
            name='bidirectional_lstm'
        )(inputs)
        
        # Attention mechanism
        attention_weights = layers.Dense(1, activation='tanh')(lstm_out)
        attention_weights = layers.Flatten()(attention_weights)
        attention_weights = layers.Activation('softmax')(attention_weights)
        attention_weights = layers.RepeatVector(self.config.hidden_units[0] * 2)(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights)
        
        # Apply attention
        attended_representation = layers.multiply([lstm_out, attention_weights])
        attended_representation = layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=1),
            name='attention_sum'
        )(attended_representation)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(attended_representation)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output
        outputs = layers.Dense(1)(x)
        
        # Create model
        self.model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='lstm_attention_predictor'
        )
        
        # Compile
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"Built LSTM-Attention model with {self.model.count_params():,} parameters")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_attention: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            use_attention: Whether to use attention mechanism
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            if use_attention:
                self.build_attention_model()
            else:
                self.build_model()
        
        # Prepare callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Add tensorboard callback if specified
        if kwargs.get('use_tensorboard', False):
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1
                )
            )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=kwargs.get('verbose', 1),
            shuffle=False  # Important for time series
        )
        
        # Store training history
        self.training_history = history.history
        self.is_trained = True
        
        # Calculate feature importance (using gradient-based method)
        self._calculate_feature_importance(X_train[:100], y_train[:100])
        
        return {
            'history': self.training_history,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history.get('val_loss', [None])[-1],
            'epochs_trained': len(history.history['loss'])
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Input sequences
            return_confidence: Whether to return prediction intervals
            
        Returns:
            Predictions and optionally confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0).flatten()
        
        if return_confidence:
            # Use MC Dropout for uncertainty estimation
            confidence_intervals = self._calculate_confidence_intervals(X)
            return predictions, confidence_intervals
        
        return predictions
    
    def _calculate_confidence_intervals(
        self,
        X: np.ndarray,
        n_iterations: int = 100,
        confidence_level: float = 0.95
    ) -> np.ndarray:
        """
        Calculate confidence intervals using MC Dropout
        
        Args:
            X: Input data
            n_iterations: Number of forward passes
            confidence_level: Confidence level for intervals
            
        Returns:
            Confidence intervals (lower, upper)
        """
        # Enable dropout during inference
        predictions = []
        
        for _ in range(n_iterations):
            # Make prediction with dropout enabled
            pred = self.model(X, training=True).numpy().flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        
        return np.stack([lower, upper], axis=1)
    
    def _calculate_feature_importance(
        self,
        X_sample: np.ndarray,
        y_sample: np.ndarray
    ) -> None:
        """
        Calculate feature importance using gradient-based method
        
        Args:
            X_sample: Sample of input data
            y_sample: Sample of target data
        """
        # Calculate gradients
        feature_importance = np.zeros(self.config.input_features)
        
        for i in range(len(X_sample)):
            x = tf.constant(X_sample[i:i+1], dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(x)
                prediction = self.model(x)
            
            # Get gradients
            gradients = tape.gradient(prediction, x)
            
            # Average absolute gradients across time steps
            importance = np.mean(np.abs(gradients.numpy()[0]), axis=0)
            feature_importance += importance
        
        # Normalize
        feature_importance = feature_importance / len(X_sample)
        feature_importance = feature_importance / feature_importance.sum()
        
        # Store as dictionary
        if self.config.feature_columns:
            self.feature_importance = {
                col: float(imp) 
                for col, imp in zip(self.config.feature_columns, feature_importance)
            }
        else:
            self.feature_importance = {
                f'feature_{i}': float(imp) 
                for i, imp in enumerate(feature_importance)
            }
    
    def predict_next(
        self,
        historical_data: pd.DataFrame,
        steps: int = 1,
        preprocessor: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Predict next values given historical data
        
        Args:
            historical_data: Historical price/feature data
            steps: Number of steps to predict ahead
            preprocessor: Data preprocessor instance
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if preprocessor is None:
            raise ValueError("Preprocessor required for live predictions")
        
        predictions = []
        prediction_dates = []
        
        # Get last sequence
        current_sequence = preprocessor.prepare_live_data(historical_data)
        
        # Make multi-step predictions
        for step in range(steps):
            # Predict next value
            pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Calculate prediction date
            last_date = historical_data.index[-1]
            if hasattr(last_date, 'freq'):
                next_date = last_date + (step + 1) * last_date.freq
            else:
                # Assume daily frequency
                next_date = last_date + pd.Timedelta(days=step+1)
            prediction_dates.append(next_date)
            
            # Update sequence for next prediction (if multi-step)
            if step < steps - 1:
                # Shift sequence and add prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                # This is simplified - in practice, you'd need to update all features
                current_sequence[0, -1, 0] = pred
        
        # Create output DataFrame
        predictions_scaled = np.array(predictions)
        
        # Inverse transform if needed
        if preprocessor.scale_target:
            predictions_original = preprocessor.inverse_transform_predictions(predictions_scaled)
        else:
            predictions_original = predictions_scaled
        
        result = pd.DataFrame({
            'prediction': predictions_original,
            'prediction_scaled': predictions_scaled
        }, index=prediction_dates)
        
        return result
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model and configuration"""
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(filepath / 'lstm_model.h5')
        
        # Save config
        self.config.save(filepath / 'config.json')
        
        # Save training history and other metadata
        metadata = {
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load model from file"""
        filepath = Path(filepath)
        
        # Load model
        self.model = keras.models.load_model(filepath / 'lstm_model.h5')
        
        # Load config
        self.config = ModelConfig.load(filepath / 'config.json')
        
        # Load metadata
        with open(filepath / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.is_trained = metadata['is_trained']
        self.training_history = metadata['training_history']
        self.feature_importance = metadata['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")\n\n# Backwards compatibility alias\nLSTMPricePredictor = LSTMPredictor\n\n__all__ = ['LSTMPricePredictor', 'LSTMPredictor']\n
