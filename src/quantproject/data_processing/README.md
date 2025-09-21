# Data Processing Module

Comprehensive data processing pipeline for financial market data, including cleaning, validation, feature engineering, and technical indicator calculation.

## Features

### 1. Data Cleaning (`data_cleaner.py`)
- **OHLCV Data Cleaning**: Handle missing values, remove invalid OHLC relationships
- **Outlier Detection**: Z-score and IQR methods for anomaly detection
- **Data Resampling**: Convert data to different time frequencies
- **Multi-Dataset Alignment**: Synchronize multiple data sources

### 2. Data Validation (`data_validator.py`)
- **Structure Validation**: Check required columns and data types
- **OHLC Relationship Checks**: Ensure high >= low, etc.
- **Missing Data Analysis**: Identify and report missing values
- **Time Gap Detection**: Find gaps in time series data
- **Volume Validation**: Check for negative or anomalous volumes
- **Quality Scoring**: Overall data quality score (0-100)

### 3. Feature Engineering (`feature_engineering.py`)
- **Price Features**: Ratios, ranges, candlestick patterns
- **Volume Features**: Volume moving averages, ratios
- **Return Features**: Simple and log returns, rolling statistics
- **Volatility Features**: Historical, Parkinson, Garman-Klass volatility
- **Time Features**: Hour, day of week, trading session indicators
- **Lag & Rolling Features**: Create lagged values and rolling window statistics
- **Feature Scaling**: Standard, MinMax, or Robust scaling

### 4. Technical Indicators (`indicators.py`)
- **Trend Indicators**: SMA, EMA, MACD
- **Momentum Indicators**: RSI, Stochastic
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, VWAP
- **Trend Strength**: ADX

### 5. Complete Pipeline (`pipeline.py`)
- **End-to-End Processing**: From raw data to ML-ready features
- **Configurable Steps**: Enable/disable any processing step
- **Training Data Preparation**: Create sequences for LSTM models
- **Validation Reports**: Comprehensive data quality reports

## Usage

### Basic Usage

```python
from quantproject.data_processing import DataProcessingPipeline

# Create pipeline
pipeline = DataProcessingPipeline()

# Process data
processed_df = pipeline.process(
    'path/to/data.csv',
    validate_first=True,
    clean_data=True,
    engineer_features=True,
    calculate_indicators=True,
    scale_features=True
)

# Get validation report
print(pipeline.get_validation_summary())
```

### Data Cleaning Only

```python
from quantproject.data_processing import DataCleaner

cleaner = DataCleaner()
clean_df = cleaner.clean_ohlcv_data(
    raw_df,
    remove_weekends=True,
    forward_fill_limit=5
)
```

### Calculate Technical Indicators

```python
from quantproject.data_processing import TechnicalIndicators

# Calculate all indicators
df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

# Calculate specific indicator
rsi = TechnicalIndicators.rsi(df['close'], period=14)
```

### Feature Engineering

```python
from quantproject.data_processing import FeatureEngineer

engineer = FeatureEngineer(scaling_method='standard')

# Engineer all features
df_features = engineer.engineer_all_features(
    df,
    feature_groups=['price', 'volume', 'returns', 'technical']
)

# Scale features
df_scaled = engineer.scale_features(df_features)
```

### Prepare Training Data

```python
# Process data for ML training
training_data = pipeline.process_for_training(
    data='path/to/data.csv',
    target_column='close',
    prediction_horizon=1,
    train_size=0.8,
    create_sequences=True,
    sequence_length=60
)

# Access training arrays
X_train = training_data['X_train']
y_train = training_data['y_train']
X_train_seq = training_data['X_train_sequences']  # For LSTM
```

## Configuration

### Cleaning Configuration
```python
cleaning_config = {
    'missing_threshold': 0.1,  # Max 10% missing data
    'outlier_std': 5,          # 5 standard deviations for outliers
}
```

### Validation Configuration
```python
validation_config = {
    'check_gaps': True,
    'check_outliers': True,
    'check_volumes': True
}
```

## Feature Groups

When engineering features, you can specify which groups to include:
- `'price'`: Price-based features (ratios, ranges, patterns)
- `'volume'`: Volume-based features
- `'returns'`: Return calculations
- `'volatility'`: Volatility measures
- `'time'`: Time-based features
- `'technical'`: Technical indicators

## Output Format

The processed DataFrame will contain:
- Original OHLCV columns (cleaned)
- Technical indicators (if enabled)
- Engineered features (if enabled)
- All features properly scaled (if enabled)

## Validation Report Example

```
==================================================
DATA VALIDATION REPORT
==================================================
Total Rows: 10000
Valid: Yes
Quality Score: 92.50/100

WARNINGS:
  - Found 15 time gaps in data
  - Found 23 outliers in close returns

STATISTICS:
  Missing Data:
    - volume: 0.50%
  Outliers:
    - open: 18 outliers
    - close: 23 outliers
  Time Gaps: 15 found
==================================================
```

## Best Practices

1. **Always validate first**: Run validation to understand data quality
2. **Clean before features**: Clean data before engineering features
3. **Scale last**: Scale features as the final step
4. **Check validation report**: Review issues and warnings
5. **Save processed data**: Save processed data for reproducibility

## Next Steps

This module provides clean, validated, and feature-rich data ready for:
- Machine learning model training
- Backtesting strategies
- Statistical analysis
- Visualization