"""
Machine Learning Models for Financial Prediction

This module provides:
- LSTM trend prediction models
- Feature extraction pipelines
- Model training and evaluation utilities
"""

from .base_model import BasePredictor, ModelConfig
from .lstm_predictor import LSTMPredictor
from .data_preprocessor import TimeSeriesPreprocessor
from .model_trainer import ModelTrainer

__all__ = [
    "BasePredictor",
    "ModelConfig",
    "LSTMPredictor",
    "TimeSeriesPreprocessor",
    "ModelTrainer",
]
