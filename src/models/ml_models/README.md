# LSTM Trend Prediction Models

Advanced LSTM-based models for financial time series prediction, providing trend forecasting capabilities for the quantitative trading system.

## Features

### Core Components

1. **LSTM Predictor** (`lstm_predictor.py`)
   - Standard LSTM architecture with dropout and regularization
   - Bidirectional LSTM with attention mechanism
   - Multi-step ahead predictions
   - Uncertainty estimation via MC Dropout
   - Feature importance calculation

2. **Data Preprocessor** (`data_preprocessor.py`)
   - Time series sequence generation
   - Multiple scaling methods (Standard, MinMax, Robust)
   - Technical indicator calculation
   - Train/test split for time series
   - Live data preparation

3. **Model Trainer** (`model_trainer.py`)
   - Automated training pipeline
   - Time series cross-validation
   - Hyperparameter optimization
   - Performance visualization
   - Model persistence

4. **Feature Extractor** (`feature_extractor.py`)
   - LSTM-based feature generation
   - Multi-horizon predictions as features
   - Confidence intervals
   - Trend analysis features
   - Integration with RL pipeline

## Quick Start

### Basic Training

```python
from src.models.ml_models import LSTMPredictor, TimeSeriesPreprocessor, ModelTrainer, ModelConfig

# Configure model
config = ModelConfig(
    input_features=10,
    sequence_length=60,
    hidden_units=[128, 64, 32],
    dropout_rate=0.2,
    learning_rate=0.001
)

# Create components
model = LSTMPredictor(config)
preprocessor = TimeSeriesPreprocessor(
    sequence_length=60,
    prediction_horizon=1,
    feature_columns=['close', 'volume', 'rsi', 'macd']
)

# Train model
trainer = ModelTrainer(model, preprocessor)
results = trainer.train_model(
    data,
    test_size=0.2,
    validation_split=0.1
)
```

### Feature Extraction

```python
from src.models.ml_models import LSTMFeatureExtractor

# Create extractor with pre-trained models
extractor = LSTMFeatureExtractor(
    model_path='./models/lstm',
    prediction_horizons=[1, 5, 20]
)

# Load models
extractor.load_models()

# Extract features
enhanced_data = extractor.extract_features(
    data,
    include_confidence=True,
    include_trend=True
)
```

### Live Predictions

```python
# Get live predictions
predictions = extractor.get_live_predictions(
    historical_data=recent_data,
    current_features={'volume': 1000000, 'rsi': 45}
)

# Access predictions
for horizon, pred in predictions.items():
    print(f"{horizon}: {pred['expected_return']:.2%}")
```

## Model Architectures

### Standard LSTM
```
Input (batch, 60, features)
    ↓
LSTM Layer 1 (128 units)
    ↓
BatchNorm + Dropout
    ↓
LSTM Layer 2 (64 units)
    ↓
BatchNorm + Dropout
    ↓
LSTM Layer 3 (32 units)
    ↓
Dense (32) + ReLU
    ↓
Output (batch, 1)
```

### LSTM with Attention
```
Input (batch, 60, features)
    ↓
Bidirectional LSTM
    ↓
Attention Mechanism
    ↓
Weighted Sum
    ↓
Dense Layers
    ↓
Output (batch, 1)
```

## Training Pipeline

### 1. Data Preparation
- Add technical indicators
- Handle missing values
- Scale features
- Create sequences

### 2. Model Training
- Early stopping
- Learning rate reduction
- Model checkpointing
- Validation monitoring

### 3. Evaluation
- MSE, RMSE, MAE
- R-squared score
- Directional accuracy
- Trading performance

### 4. Feature Generation
- Multi-horizon predictions
- Confidence intervals
- Trend indicators
- Return forecasts

## Configuration Options

### Model Config
```python
config = ModelConfig(
    # Architecture
    model_type='lstm',          # or 'lstm_attention'
    hidden_units=[128, 64, 32], # Layer sizes
    dropout_rate=0.2,           # Dropout probability
    
    # Training
    batch_size=32,
    epochs=100,
    learning_rate=0.001,
    early_stopping_patience=10,
    
    # Data
    sequence_length=60,         # Input sequence length
    prediction_horizon=1,       # Steps ahead to predict
    feature_columns=['close', 'volume', 'rsi'],
    target_column='close'
)
```

### Preprocessor Config
```python
preprocessor = TimeSeriesPreprocessor(
    sequence_length=60,
    prediction_horizon=1,
    scaling_method='standard',  # 'standard', 'minmax', 'robust'
    scale_target=True,         # Scale target variable
    feature_columns=None       # Auto-detect if None
)
```

## Performance Optimization

### Training Tips
1. **Sequence Length**: 60-120 for daily data
2. **Batch Size**: 32-64 for stable training
3. **Learning Rate**: Start with 0.001, reduce on plateau
4. **Dropout**: 0.2-0.3 to prevent overfitting
5. **Early Stopping**: Monitor validation loss

### Feature Engineering
1. **Price Features**: Returns, log returns, ratios
2. **Volume Features**: Volume ratios, OBV
3. **Technical Indicators**: RSI, MACD, Bollinger Bands
4. **Time Features**: Day of week, month effects

### Model Selection
- **Standard LSTM**: General trend prediction
- **LSTM + Attention**: When interpretability matters
- **Multi-Horizon**: For different trading strategies

## Integration with RL

The LSTM predictions serve as "sensory" inputs for RL agents:

```python
# In RL environment
lstm_features = extractor.extract_features(market_data)

# Add to state space
state = np.concatenate([
    market_features,
    lstm_features[['lstm_pred_1', 'lstm_pred_5', 'lstm_pred_20']].values
])
```

## Monitoring and Debugging

### Training Monitoring
- Loss curves (training_history.png)
- Prediction plots (predictions.png)
- Feature importance scores
- Cross-validation results

### Model Diagnostics
```python
# Check feature importance
importance = model.get_feature_importance()

# Analyze predictions
predictions, confidence = model.predict(X_test, return_confidence=True)

# Evaluate performance
metrics = model.evaluate(X_test, y_test)
```

## Best Practices

1. **Data Quality**
   - Clean outliers before training
   - Handle missing values appropriately
   - Ensure sufficient history (>1000 samples)

2. **Validation Strategy**
   - Use time series split
   - Keep test set truly out-of-sample
   - Monitor for data leakage

3. **Production Deployment**
   - Retrain periodically
   - Monitor prediction drift
   - Implement fallback strategies

## Next Steps

This LSTM module provides trend prediction capabilities that will be integrated with:
- FinBERT sentiment analysis (Phase 2.2)
- Visualization dashboard (Phase 2.3)
- RL decision making (Phase 3)

The predictions serve as key features for the RL agent's state space, enabling informed trading decisions based on anticipated market movements.