"""
Example usage of LSTM trend prediction models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from lstm_predictor import LSTMPredictor
from data_preprocessor import TimeSeriesPreprocessor
from model_trainer import ModelTrainer
from base_model import ModelConfig
from feature_extractor import LSTMFeatureExtractor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_data(days: int = 1000) -> pd.DataFrame:
    """Create sample financial data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate synthetic price data with trend and noise
    np.random.seed(42)
    trend = np.linspace(100, 150, days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = np.random.normal(0, 5, days)
    
    prices = trend + seasonal + noise
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    data['volume'] = np.random.randint(1000000, 5000000, days)
    
    # Add some technical indicators
    data['returns'] = data['close'].pct_change()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = calculate_rsi(data['close'])
    
    return data.dropna()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def example_basic_lstm_training():
    """Basic LSTM training example"""
    print("\n" + "="*60)
    print("BASIC LSTM TRAINING EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=1000)
    print(f"Created sample data with {len(data)} days")
    
    # Configure model
    config = ModelConfig(
        model_type='lstm',
        input_features=4,  # close, volume, returns, rsi
        sequence_length=60,
        hidden_units=[64, 32],
        dropout_rate=0.2,
        batch_size=32,
        epochs=50,
        learning_rate=0.001,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close',
        prediction_horizon=1
    )
    
    # Create model and preprocessor
    model = LSTMPredictor(config)
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=60,
        prediction_horizon=1,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close',
        scaling_method='standard'
    )
    
    # Train model
    trainer = ModelTrainer(model, preprocessor, output_dir='./output/basic_lstm')
    results = trainer.train_model(
        data,
        test_size=0.2,
        validation_split=0.1,
        plot_results=True,
        save_model=True
    )
    
    # Print results
    print("\nTraining Results:")
    print(f"Final test RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"Final test R²: {results['test_metrics']['r2']:.4f}")
    print(f"Directional accuracy: {results['test_metrics'].get('directional_accuracy', 0):.2%}")


def example_multi_horizon_prediction():
    """Example of multi-horizon LSTM predictions"""
    print("\n" + "="*60)
    print("MULTI-HORIZON LSTM PREDICTION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=1000)
    
    # Train models for different horizons
    horizons = [1, 5, 20]  # 1-day, 5-day, 20-day predictions
    
    for horizon in horizons:
        print(f"\nTraining {horizon}-day prediction model...")
        
        # Configure model
        config = ModelConfig(
            model_type='lstm',
            input_features=4,
            sequence_length=60,
            hidden_units=[128, 64, 32],
            dropout_rate=0.3,
            batch_size=32,
            epochs=30,  # Reduced for demo
            learning_rate=0.001,
            feature_columns=['close', 'volume', 'returns', 'rsi'],
            target_column='close',
            prediction_horizon=horizon
        )
        
        # Create and train model
        model = LSTMPredictor(config)
        preprocessor = TimeSeriesPreprocessor(
            sequence_length=60,
            prediction_horizon=horizon,
            feature_columns=['close', 'volume', 'returns', 'rsi'],
            target_column='close'
        )
        
        trainer = ModelTrainer(
            model, 
            preprocessor, 
            output_dir=f'./output/lstm_horizon_{horizon}'
        )
        
        results = trainer.train_model(
            data,
            test_size=0.2,
            validation_split=0.1,
            plot_results=False,
            save_model=True
        )
        
        print(f"{horizon}-day prediction - RMSE: {results['test_metrics']['rmse']:.4f}, "
              f"R²: {results['test_metrics']['r2']:.4f}")


def example_lstm_with_attention():
    """Example using LSTM with attention mechanism"""
    print("\n" + "="*60)
    print("LSTM WITH ATTENTION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=1000)
    
    # Configure model
    config = ModelConfig(
        model_type='lstm_attention',
        input_features=4,
        sequence_length=60,
        hidden_units=[128],  # Only first layer used for bidirectional
        dropout_rate=0.2,
        batch_size=32,
        epochs=50,
        learning_rate=0.001,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close',
        prediction_horizon=1
    )
    
    # Create model and preprocessor
    model = LSTMPredictor(config)
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=60,
        prediction_horizon=1,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close'
    )
    
    # Prepare data
    split_data = preprocessor.create_train_test_split(data, test_size=0.2)
    
    # Train with attention
    print("Training LSTM with attention mechanism...")
    history = model.train(
        split_data['X_train'],
        split_data['y_train'],
        split_data['X_test'],
        split_data['y_test'],
        use_attention=True
    )
    
    # Evaluate
    test_metrics = model.evaluate(split_data['X_test'], split_data['y_test'])
    print(f"\nTest RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")


def example_feature_extraction():
    """Example of using LSTM for feature extraction"""
    print("\n" + "="*60)
    print("LSTM FEATURE EXTRACTION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=500)
    
    # Create feature extractor
    extractor = LSTMFeatureExtractor(
        prediction_horizons=[1, 5, 20],
        feature_prefix='lstm'
    )
    
    # Train models for feature extraction
    print("Training LSTM models for feature extraction...")
    training_results = extractor.train_models(
        data,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close',
        sequence_length=60,
        plot_results=False,
        save_model=True
    )
    
    # Extract features
    print("\nExtracting LSTM-based features...")
    features = extractor.extract_features(
        data,
        include_confidence=True,
        include_trend=True
    )
    
    # Show extracted features
    new_features = [col for col in features.columns if col.startswith('lstm')]
    print(f"\nExtracted {len(new_features)} new features:")
    for feat in new_features[:10]:  # Show first 10
        print(f"  - {feat}")
    
    # Get live predictions
    print("\nGetting live predictions...")
    live_predictions = extractor.get_live_predictions(data.tail(100))
    
    for horizon, pred_data in live_predictions.items():
        print(f"\n{horizon}:")
        print(f"  Current price: ${pred_data['current_price']:.2f}")
        print(f"  Prediction: ${pred_data['prediction']:.2f}")
        print(f"  Expected return: {pred_data['expected_return']*100:.2f}%")


def example_cross_validation():
    """Example of time series cross-validation"""
    print("\n" + "="*60)
    print("TIME SERIES CROSS-VALIDATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=1000)
    
    # Configure model
    config = ModelConfig(
        model_type='lstm',
        input_features=4,
        sequence_length=60,
        hidden_units=[64, 32],
        dropout_rate=0.2,
        batch_size=32,
        epochs=20,  # Reduced for CV
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close'
    )
    
    # Create model and preprocessor
    model = LSTMPredictor(config)
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=60,
        prediction_horizon=1,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close'
    )
    
    # Perform cross-validation
    trainer = ModelTrainer(model, preprocessor, output_dir='./output/cv_results')
    cv_results = trainer.cross_validate(data, n_splits=5, gap=20)
    
    # Print CV results
    print("\nCross-Validation Results:")
    print(f"Mean RMSE: {cv_results['mean_metrics']['rmse']:.4f} "
          f"(±{cv_results['std_metrics']['rmse']:.4f})")
    print(f"Mean R²: {cv_results['mean_metrics']['r2']:.4f} "
          f"(±{cv_results['std_metrics']['r2']:.4f})")
    print(f"Mean Directional Accuracy: {cv_results['mean_metrics']['directional_accuracy']:.2%}")


def example_live_prediction():
    """Example of making live predictions"""
    print("\n" + "="*60)
    print("LIVE PREDICTION EXAMPLE")
    print("="*60)
    
    # Create sample data
    historical_data = create_sample_data(days=500)
    
    # Train a model (in practice, you'd load a pre-trained model)
    config = ModelConfig(
        model_type='lstm',
        input_features=4,
        sequence_length=60,
        hidden_units=[64, 32],
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close',
        prediction_horizon=1
    )
    
    model = LSTMPredictor(config)
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=60,
        prediction_horizon=1,
        feature_columns=['close', 'volume', 'returns', 'rsi'],
        target_column='close'
    )
    
    # Quick training for demo
    split_data = preprocessor.create_train_test_split(historical_data, test_size=0.2)
    model.build_model()
    model.train(
        split_data['X_train'][:100],  # Small subset for quick demo
        split_data['y_train'][:100],
        epochs=10,
        verbose=0
    )
    
    # Make live prediction
    print("\nMaking live prediction...")
    next_predictions = model.predict_next(
        historical_data,
        steps=5,
        preprocessor=preprocessor
    )
    
    print("\nNext 5 days predictions:")
    print(next_predictions)
    
    # Calculate trading signal
    current_price = historical_data['close'].iloc[-1]
    next_price = next_predictions['prediction'].iloc[0]
    signal_strength = (next_price - current_price) / current_price
    
    print(f"\nTrading Signal:")
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted price: ${next_price:.2f}")
    print(f"Signal: {'BUY' if signal_strength > 0.01 else 'SELL' if signal_strength < -0.01 else 'HOLD'}")
    print(f"Strength: {abs(signal_strength)*100:.2f}%")


if __name__ == "__main__":
    # Run examples
    example_basic_lstm_training()
    example_multi_horizon_prediction()
    example_lstm_with_attention()
    example_feature_extraction()
    example_cross_validation()
    example_live_prediction()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)