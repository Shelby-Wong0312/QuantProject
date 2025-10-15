"""
Data loader for integrating with existing data pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Import from existing modules
    from data_processing.data_cleaner import DataCleaner
    from data_processing.feature_engineering import FeatureEngineer
    from models.ml_models.lstm_predictor import LSTMPredictor
    from models.sentiment.finbert_analyzer import FinBERTAnalyzer
    from models.sentiment.sentiment_scorer import SentimentScorer
    from backtesting.portfolio import Portfolio
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Using mock data instead.")

logger = logging.getLogger(__name__)


class DashboardDataLoader:
    """Load and prepare data for dashboard visualization"""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing saved data
        """
        self.data_dir = data_dir or Path("./data")
        self.cache = {}

        # Initialize components if available
        try:
            self.data_cleaner = DataCleaner()
            self.feature_engineer = FeatureEngineer()
        except:
            self.data_cleaner = None
            self.feature_engineer = None

    def load_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Load market data for a symbol

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Try to load from saved data
            file_path = self.data_dir / f"{symbol}_ohlcv.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=["datetime"])
                df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

                # Clean data if cleaner available
                if self.data_cleaner:
                    df = self.data_cleaner.clean_ohlcv_data(df)

                self.cache[cache_key] = df
                return df
            else:
                # Return mock data if file doesn't exist
                return self._generate_mock_market_data(symbol, start_date, end_date)

        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return self._generate_mock_market_data(symbol, start_date, end_date)

    def load_lstm_predictions(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Load LSTM predictions

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with prediction DataFrames
        """
        try:
            # Try to load saved predictions
            predictions_file = self.data_dir / f"{symbol}_lstm_predictions.csv"
            if predictions_file.exists():
                df = pd.read_csv(predictions_file, parse_dates=["datetime"])
                df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

                return {
                    "1d": df[["datetime", "pred_1d", "confidence_1d"]],
                    "5d": df[["datetime", "pred_5d", "confidence_5d"]],
                    "20d": df[["datetime", "pred_20d", "confidence_20d"]],
                }
            else:
                return self._generate_mock_predictions(symbol, start_date, end_date)

        except Exception as e:
            logger.error(f"Error loading LSTM predictions: {str(e)}")
            return self._generate_mock_predictions(symbol, start_date, end_date)

    def load_sentiment_data(self, symbol: str, hours: int = 24) -> Dict[str, any]:
        """
        Load sentiment analysis data

        Args:
            symbol: Stock symbol
            hours: Hours of historical data

        Returns:
            Dictionary with sentiment data
        """
        try:
            # Try to load saved sentiment data
            sentiment_file = self.data_dir / f"{symbol}_sentiment.csv"
            if sentiment_file.exists():
                df = pd.read_csv(sentiment_file, parse_dates=["timestamp"])

                # Filter recent data
                cutoff = datetime.now() - timedelta(hours=hours)
                df = df[df["timestamp"] >= cutoff]

                return {
                    "current_score": df.iloc[-1]["sentiment_score"] if len(df) > 0 else 0,
                    "historical": df[["timestamp", "sentiment_score"]],
                    "news": df[["title", "sentiment", "confidence", "timestamp"]],
                }
            else:
                return self._generate_mock_sentiment_data(symbol, hours)

        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
            return self._generate_mock_sentiment_data(symbol, hours)

    def load_backtest_signals(self, symbol: str, strategy: str = "default") -> pd.DataFrame:
        """
        Load backtest trading signals

        Args:
            symbol: Stock symbol
            strategy: Strategy name

        Returns:
            DataFrame with trading signals
        """
        try:
            # Try to load saved signals
            signals_file = self.data_dir / f"{symbol}_{strategy}_signals.csv"
            if signals_file.exists():
                df = pd.read_csv(signals_file, parse_dates=["datetime"])
                return df
            else:
                return self._generate_mock_signals(symbol)

        except Exception as e:
            logger.error(f"Error loading backtest signals: {str(e)}")
            return self._generate_mock_signals(symbol)

    def calculate_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators
        """
        if self.feature_engineer:
            try:
                return self.feature_engineer.engineer_features(market_data)
            except:
                pass

        # Fallback to simple calculations
        df = market_data.copy()

        # Simple moving averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df

    def _generate_mock_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate mock market data"""
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Generate realistic price movement
        base_price = 100
        returns = np.random.normal(0.0002, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.lognormal(15, 1))

            data.append(
                {
                    "datetime": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        return pd.DataFrame(data)

    def _generate_mock_predictions(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Generate mock LSTM predictions"""
        market_data = self.load_market_data(symbol, start_date, end_date)

        predictions = {}
        for horizon, noise_level in [("1d", 0.01), ("5d", 0.03), ("20d", 0.05)]:
            pred = market_data["close"] * (1 + np.random.normal(0, noise_level, len(market_data)))
            confidence = np.random.uniform(0.6, 0.9, len(market_data))

            predictions[horizon] = pd.DataFrame(
                {"datetime": market_data["datetime"], "prediction": pred, "confidence": confidence}
            )

        return predictions

    def _generate_mock_sentiment_data(self, symbol: str, hours: int) -> Dict[str, any]:
        """Generate mock sentiment data"""
        # Historical sentiment
        times = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
        scores = np.random.normal(0, 0.3, hours)
        scores = np.clip(scores, -1, 1)

        historical = pd.DataFrame({"timestamp": times, "sentiment_score": scores})

        # Mock news
        news_data = []
        for i in range(5):
            sentiment = np.random.choice(["positive", "negative", "neutral"])
            confidence = np.random.uniform(0.7, 0.95)

            news_data.append(
                {
                    "title": f"Sample news about {symbol} #{i+1}",
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                }
            )

        news = pd.DataFrame(news_data)

        return {"current_score": scores[-1], "historical": historical, "news": news}

    def _generate_mock_signals(self, symbol: str) -> pd.DataFrame:
        """Generate mock trading signals"""
        # Use recent market data
        market_data = self.load_market_data(
            symbol, datetime.now() - timedelta(days=30), datetime.now()
        )

        signals = []
        for i in range(5, len(market_data), 10):
            signal_type = np.random.choice(["BUY", "SELL"])
            signals.append(
                {
                    "datetime": market_data.iloc[i]["datetime"],
                    "signal": signal_type,
                    "price": market_data.iloc[i]["close"],
                    "confidence": np.random.uniform(0.6, 0.9),
                }
            )

        return pd.DataFrame(signals)
