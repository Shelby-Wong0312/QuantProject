"""
State space processor for RL trading environment
Integrates LSTM predictions and FinBERT sentiment analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class StateConfig:
    """Configuration for state space"""
    # Market data features
    price_features: List[str] = None
    volume_features: List[str] = None
    technical_indicators: List[str] = None
    
    # LSTM features
    lstm_horizons: List[int] = None  # [1, 5, 20]
    lstm_confidence: bool = True
    
    # Sentiment features
    sentiment_window: int = 24  # hours
    sentiment_aggregations: List[str] = None  # ['current', 'mean', 'trend']
    
    # Portfolio features
    include_position: bool = True
    include_pnl: bool = True
    include_cash: bool = True
    
    # Time features
    include_time_features: bool = True
    
    def __post_init__(self):
        if self.price_features is None:
            self.price_features = ['close', 'high', 'low', 'open']
        if self.volume_features is None:
            self.volume_features = ['volume', 'volume_ma']
        if self.technical_indicators is None:
            self.technical_indicators = ['rsi', 'macd', 'bb_position', 'atr']
        if self.lstm_horizons is None:
            self.lstm_horizons = [1, 5, 20]
        if self.sentiment_aggregations is None:
            self.sentiment_aggregations = ['current', 'mean', 'trend', 'volatility']


class StateProcessor:
    """
    Process and normalize state space for RL environment
    """
    
    def __init__(self, config: Optional[StateConfig] = None):
        """
        Initialize state processor
        
        Args:
            config: State configuration
        """
        self.config = config or StateConfig()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Feature indices for quick access
        self.feature_indices = {}
        
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit the state processor on historical data
        
        Args:
            historical_data: Historical market data with all features
        """
        logger.info("Fitting state processor on historical data")
        
        # Extract all features
        features = self._extract_all_features(historical_data)
        
        # Fit scaler
        self.scaler.fit(features)
        self.is_fitted = True
        
        # Store feature names and indices
        self.feature_names = list(features.columns)
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        
        logger.info(f"State processor fitted with {len(self.feature_names)} features")
        
    def process_state(
        self,
        market_data: pd.DataFrame,
        lstm_predictions: Dict[int, Dict[str, float]],
        sentiment_data: Dict[str, Any],
        portfolio_state: Dict[str, float],
        current_time: pd.Timestamp
    ) -> np.ndarray:
        """
        Process raw data into normalized state vector
        
        Args:
            market_data: Recent market data
            lstm_predictions: LSTM predictions for different horizons
            sentiment_data: FinBERT sentiment analysis results
            portfolio_state: Current portfolio state
            current_time: Current timestamp
            
        Returns:
            Normalized state vector
        """
        if not self.is_fitted:
            raise ValueError("StateProcessor must be fitted before processing states")
        
        features = {}
        
        # 1. Market data features
        features.update(self._process_market_features(market_data))
        
        # 2. LSTM prediction features
        features.update(self._process_lstm_features(lstm_predictions))
        
        # 3. Sentiment features
        features.update(self._process_sentiment_features(sentiment_data))
        
        # 4. Portfolio features
        if self.config.include_position:
            features.update(self._process_portfolio_features(portfolio_state))
        
        # 5. Time features
        if self.config.include_time_features:
            features.update(self._process_time_features(current_time))
        
        # Convert to DataFrame for scaling
        features_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training
        features_df = features_df[self.feature_names]
        
        # Scale features
        state_vector = self.scaler.transform(features_df).flatten()
        
        return state_vector
    
    def _process_market_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Process market data features"""
        features = {}
        current = market_data.iloc[-1]
        
        # Price features
        for feat in self.config.price_features:
            if feat in market_data.columns:
                features[f'market_{feat}'] = current[feat]
                
                # Price changes
                if len(market_data) > 1:
                    features[f'market_{feat}_change'] = (
                        current[feat] / market_data.iloc[-2][feat] - 1
                    )
        
        # Volume features
        for feat in self.config.volume_features:
            if feat in market_data.columns:
                features[f'market_{feat}'] = current[feat]
        
        # Technical indicators
        for indicator in self.config.technical_indicators:
            if indicator in market_data.columns:
                features[f'tech_{indicator}'] = current[indicator]
        
        # Additional derived features
        if 'high' in market_data.columns and 'low' in market_data.columns:
            features['market_range'] = (current['high'] - current['low']) / current['close']
        
        return features
    
    def _process_lstm_features(self, lstm_predictions: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Process LSTM prediction features"""
        features = {}
        
        for horizon in self.config.lstm_horizons:
            if horizon in lstm_predictions:
                pred_data = lstm_predictions[horizon]
                
                # Prediction value
                features[f'lstm_pred_{horizon}d'] = pred_data.get('prediction', 0)
                
                # Prediction change from current price
                if 'current_price' in pred_data:
                    features[f'lstm_return_{horizon}d'] = (
                        pred_data['prediction'] / pred_data['current_price'] - 1
                    )
                
                # Confidence score
                if self.config.lstm_confidence:
                    features[f'lstm_conf_{horizon}d'] = pred_data.get('confidence', 0.5)
        
        # Cross-horizon features
        if 1 in lstm_predictions and 5 in lstm_predictions:
            features['lstm_trend_short'] = (
                lstm_predictions[5]['prediction'] - lstm_predictions[1]['prediction']
            ) / lstm_predictions[1]['prediction']
        
        if 5 in lstm_predictions and 20 in lstm_predictions:
            features['lstm_trend_long'] = (
                lstm_predictions[20]['prediction'] - lstm_predictions[5]['prediction']
            ) / lstm_predictions[5]['prediction']
        
        return features
    
    def _process_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """Process sentiment analysis features"""
        features = {}
        
        # Current sentiment score
        if 'current' in self.config.sentiment_aggregations:
            features['sentiment_current'] = sentiment_data.get('current_score', 0)
        
        # Sentiment statistics
        if 'historical' in sentiment_data:
            hist_scores = sentiment_data['historical']
            
            if 'mean' in self.config.sentiment_aggregations:
                features['sentiment_mean'] = np.mean(hist_scores)
            
            if 'trend' in self.config.sentiment_aggregations:
                if len(hist_scores) > 1:
                    # Simple linear trend
                    x = np.arange(len(hist_scores))
                    slope = np.polyfit(x, hist_scores, 1)[0]
                    features['sentiment_trend'] = slope
                else:
                    features['sentiment_trend'] = 0
            
            if 'volatility' in self.config.sentiment_aggregations:
                features['sentiment_volatility'] = np.std(hist_scores)
        
        # News volume
        features['sentiment_news_count'] = sentiment_data.get('news_count', 0)
        
        # Sentiment categories
        features['sentiment_positive_ratio'] = sentiment_data.get('positive_ratio', 0.33)
        features['sentiment_negative_ratio'] = sentiment_data.get('negative_ratio', 0.33)
        
        return features
    
    def _process_portfolio_features(self, portfolio_state: Dict[str, float]) -> Dict[str, float]:
        """Process portfolio state features"""
        features = {}
        
        # Position information
        features['portfolio_position'] = portfolio_state.get('position', 0)
        features['portfolio_position_pct'] = portfolio_state.get('position_pct', 0)
        
        # P&L information
        if self.config.include_pnl:
            features['portfolio_unrealized_pnl'] = portfolio_state.get('unrealized_pnl', 0)
            features['portfolio_realized_pnl'] = portfolio_state.get('realized_pnl', 0)
            features['portfolio_total_pnl'] = (
                features['portfolio_unrealized_pnl'] + features['portfolio_realized_pnl']
            )
        
        # Cash and buying power
        if self.config.include_cash:
            features['portfolio_cash'] = portfolio_state.get('cash', 0)
            features['portfolio_buying_power'] = portfolio_state.get('buying_power', 0)
        
        # Risk metrics
        features['portfolio_exposure'] = abs(features['portfolio_position_pct'])
        
        return features
    
    def _process_time_features(self, current_time: pd.Timestamp) -> Dict[str, float]:
        """Process time-based features"""
        features = {}
        
        # Time of day (normalized to 0-1)
        features['time_hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
        features['time_hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
        
        # Day of week (normalized)
        features['time_dow'] = current_time.dayofweek / 6
        
        # Trading session progress (assuming 9:30 AM - 4:00 PM)
        market_open = current_time.replace(hour=9, minute=30)
        market_close = current_time.replace(hour=16, minute=0)
        
        if market_open <= current_time <= market_close:
            session_progress = (current_time - market_open).seconds / (6.5 * 3600)
            features['time_session_progress'] = session_progress
        else:
            features['time_session_progress'] = 0
        
        # Is market open
        features['time_market_open'] = float(market_open <= current_time <= market_close)
        
        return features
    
    def _extract_all_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from historical data for fitting"""
        all_features = []
        
        for i in range(len(historical_data)):
            if i < 20:  # Need enough history
                continue
            
            # Simulate feature extraction
            features = {}
            
            # Market features
            current = historical_data.iloc[i]
            for col in self.config.price_features + self.config.volume_features:
                if col in historical_data.columns:
                    features[f'market_{col}'] = current[col]
                    if i > 0:
                        features[f'market_{col}_change'] = (
                            current[col] / historical_data.iloc[i-1][col] - 1
                        )
            
            # Technical indicators
            for indicator in self.config.technical_indicators:
                features[f'tech_{indicator}'] = np.random.randn()
            
            # LSTM features (simulated)
            for horizon in self.config.lstm_horizons:
                features[f'lstm_pred_{horizon}d'] = current['close'] * (1 + np.random.randn() * 0.01)
                features[f'lstm_return_{horizon}d'] = np.random.randn() * 0.01
                features[f'lstm_conf_{horizon}d'] = np.random.uniform(0.5, 0.9)
            
            # Sentiment features (simulated)
            features['sentiment_current'] = np.random.uniform(-1, 1)
            features['sentiment_mean'] = np.random.uniform(-1, 1)
            features['sentiment_trend'] = np.random.uniform(-0.1, 0.1)
            features['sentiment_volatility'] = np.random.uniform(0, 0.5)
            features['sentiment_news_count'] = np.random.randint(0, 10)
            features['sentiment_positive_ratio'] = np.random.uniform(0, 1)
            features['sentiment_negative_ratio'] = np.random.uniform(0, 1)
            
            # Portfolio features
            features['portfolio_position'] = np.random.uniform(-1, 1)
            features['portfolio_position_pct'] = np.random.uniform(-1, 1)
            features['portfolio_unrealized_pnl'] = np.random.randn() * 100
            features['portfolio_realized_pnl'] = np.random.randn() * 100
            features['portfolio_total_pnl'] = features['portfolio_unrealized_pnl'] + features['portfolio_realized_pnl']
            features['portfolio_cash'] = 10000 + np.random.randn() * 1000
            features['portfolio_buying_power'] = features['portfolio_cash']
            features['portfolio_exposure'] = abs(features['portfolio_position_pct'])
            
            # Time features
            features['time_hour_sin'] = np.random.uniform(-1, 1)
            features['time_hour_cos'] = np.random.uniform(-1, 1)
            features['time_dow'] = np.random.uniform(0, 1)
            features['time_session_progress'] = np.random.uniform(0, 1)
            features['time_market_open'] = np.random.choice([0, 1])
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def get_state_shape(self) -> Tuple[int,]:
        """Get the shape of the state vector"""
        if not self.is_fitted:
            raise ValueError("StateProcessor must be fitted before getting state shape")
        return (len(self.feature_names),)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()