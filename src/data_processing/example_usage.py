"""
Example usage of the data processing module
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import all components
from pipeline import DataProcessingPipeline
from data_cleaner import DataCleaner
from data_validator import DataValidator
from feature_engineering import FeatureEngineer
from indicators import TechnicalIndicators


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_data():
    """Create sample OHLCV data for demonstration"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    
    # Generate synthetic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    df = pd.DataFrame(index=dates)
    df['close'] = close_prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0]) + np.random.randn(len(dates)) * 0.1
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(len(dates)) * 0.2)
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(len(dates)) * 0.2)
    df['volume'] = np.random.randint(1000, 10000, len(dates))
    
    # Add some data quality issues for demonstration
    # Add missing values
    df.loc[df.sample(n=50).index, 'volume'] = np.nan
    
    # Add outliers
    df.loc[df.sample(n=10).index, 'high'] = df['high'] * 1.5
    
    # Add invalid OHLC relationships
    df.loc[df.sample(n=5).index, 'low'] = df.loc[df.sample(n=5).index, 'high'] + 1
    
    return df


def example_complete_pipeline():
    """Example of using the complete processing pipeline"""
    print("\n" + "="*60)
    print("COMPLETE PIPELINE EXAMPLE")
    print("="*60)
    
    # Create sample data
    raw_data = create_sample_data()
    print(f"Created sample data with {len(raw_data)} rows")
    
    # Initialize pipeline
    pipeline = DataProcessingPipeline(
        cleaning_config={'missing_threshold': 0.1, 'outlier_std': 5},
        validation_config={'check_gaps': True, 'check_outliers': True},
        scaling_method='standard'
    )
    
    # Process data
    processed_data = pipeline.process(
        raw_data,
        validate_first=True,
        clean_data=True,
        engineer_features=True,
        calculate_indicators=True,
        scale_features=True,
        feature_groups=['price', 'volume', 'returns', 'technical']
    )
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Number of features: {len(processed_data.columns)}")
    
    # Show validation report
    print("\n" + pipeline.get_validation_summary())
    
    # Show feature statistics
    feature_info = pipeline.get_feature_importance()
    print("\nTop 10 Features by Missing %:")
    print(feature_info.nlargest(10, 'missing_pct')[['feature', 'missing_pct']])


def example_data_cleaning():
    """Example of data cleaning only"""
    print("\n" + "="*60)
    print("DATA CLEANING EXAMPLE")
    print("="*60)
    
    # Create sample data with issues
    raw_data = create_sample_data()
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Clean data
    clean_data = cleaner.clean_ohlcv_data(
        raw_data,
        remove_weekends=True,
        forward_fill_limit=5
    )
    
    print(f"Original rows: {len(raw_data)}")
    print(f"Cleaned rows: {len(clean_data)}")
    print(f"Rows removed: {len(raw_data) - len(clean_data)}")
    
    # Resample to daily
    daily_data = cleaner.resample_data(clean_data, target_frequency='1D')
    print(f"\nResampled to daily: {len(daily_data)} rows")


def example_technical_indicators():
    """Example of calculating technical indicators"""
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS EXAMPLE")
    print("="*60)
    
    # Create clean sample data
    data = create_sample_data()
    cleaner = DataCleaner()
    clean_data = cleaner.clean_ohlcv_data(data)
    
    # Calculate individual indicators
    rsi = TechnicalIndicators.rsi(clean_data['close'])
    macd, signal, histogram = TechnicalIndicators.macd(clean_data['close'])
    bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(clean_data['close'])
    
    print(f"RSI - Mean: {rsi.mean():.2f}, Std: {rsi.std():.2f}")
    print(f"MACD - Mean: {macd.mean():.2f}, Std: {macd.std():.2f}")
    
    # Calculate all indicators
    data_with_indicators = TechnicalIndicators.calculate_all_indicators(clean_data)
    
    indicator_cols = [col for col in data_with_indicators.columns 
                     if col not in ['open', 'high', 'low', 'close', 'volume']]
    print(f"\nTotal indicators calculated: {len(indicator_cols)}")
    print("Indicators:", indicator_cols[:10], "...")


def example_feature_engineering():
    """Example of feature engineering"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING EXAMPLE")
    print("="*60)
    
    # Create clean sample data
    data = create_sample_data()
    cleaner = DataCleaner()
    clean_data = cleaner.clean_ohlcv_data(data)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(scaling_method='standard')
    
    # Create specific feature groups
    price_features = engineer.create_price_features(clean_data)
    volume_features = engineer.create_volume_features(clean_data)
    return_features = engineer.create_return_features(clean_data)
    
    print(f"Price features: {len([col for col in price_features.columns if col not in clean_data.columns])}")
    print(f"Volume features: {len([col for col in volume_features.columns if col not in clean_data.columns])}")
    print(f"Return features: {len([col for col in return_features.columns if col not in clean_data.columns])}")
    
    # Create all features
    all_features = engineer.engineer_all_features(clean_data)
    print(f"\nTotal features after engineering: {len(all_features.columns)}")
    
    # Scale features
    feature_cols = [col for col in all_features.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume']]
    scaled_features = engineer.scale_features(all_features[feature_cols])
    
    print(f"\nScaled features shape: {scaled_features.shape}")
    print(f"Mean of scaled features: {scaled_features.mean().mean():.4f}")
    print(f"Std of scaled features: {scaled_features.std().mean():.4f}")


def example_training_preparation():
    """Example of preparing data for model training"""
    print("\n" + "="*60)
    print("TRAINING PREPARATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    raw_data = create_sample_data()
    
    # Initialize pipeline
    pipeline = DataProcessingPipeline()
    
    # Prepare training data
    training_data = pipeline.process_for_training(
        raw_data,
        target_column='close',
        prediction_horizon=1,  # Predict 1 period ahead
        train_size=0.8,
        create_sequences=True,
        sequence_length=24  # 24 hours of history
    )
    
    print(f"Training samples: {len(training_data['X_train'])}")
    print(f"Test samples: {len(training_data['X_test'])}")
    print(f"Number of features: {training_data['X_train'].shape[1]}")
    
    if 'X_train_sequences' in training_data:
        print(f"\nSequence shape: {training_data['X_train_sequences'].shape}")
        print(f"- Samples: {training_data['X_train_sequences'].shape[0]}")
        print(f"- Timesteps: {training_data['X_train_sequences'].shape[1]}")
        print(f"- Features: {training_data['X_train_sequences'].shape[2]}")


def example_data_validation():
    """Example of data validation"""
    print("\n" + "="*60)
    print("DATA VALIDATION EXAMPLE")
    print("="*60)
    
    # Create sample data with various issues
    data = create_sample_data()
    
    # Add more issues
    data.loc[data.sample(n=100).index, 'close'] = np.nan
    data.loc[data.sample(n=20).index, 'volume'] = -1000
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate data
    is_valid, report = validator.validate_ohlcv(data)
    
    print(f"Data is valid: {is_valid}")
    print(f"Quality score: {report['quality_score']:.2f}/100")
    print(f"Number of issues: {len(report['issues'])}")
    print(f"Number of warnings: {len(report['warnings'])}")
    
    # Print detailed report
    print("\n" + validator.generate_validation_report())


if __name__ == "__main__":
    # Run all examples
    print("DATA PROCESSING MODULE EXAMPLES")
    print("="*60)
    
    # Run examples
    example_complete_pipeline()
    example_data_cleaning()
    example_technical_indicators()
    example_feature_engineering()
    example_training_preparation()
    example_data_validation()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)