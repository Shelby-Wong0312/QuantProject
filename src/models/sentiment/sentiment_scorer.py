"""
Sentiment scoring and aggregation system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
import json


logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score data structure"""

    symbol: str
    timestamp: datetime
    score: float  # Normalized score (-1 to 1)
    confidence: float  # Confidence level (0 to 1)
    volume: int  # Number of articles
    components: Dict[str, float]  # Component scores
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "confidence": self.confidence,
            "volume": self.volume,
            "components": self.components,
            "metadata": self.metadata,
        }


class SentimentScorer:
    """
    Advanced sentiment scoring system with multiple aggregation methods
    """

    def __init__(
        self,
        decay_factor: float = 0.9,
        volume_weight: float = 0.3,
        confidence_threshold: float = 0.5,
        lookback_days: int = 7,
    ):
        """
        Initialize sentiment scorer

        Args:
            decay_factor: Time decay factor for older news
            volume_weight: Weight for news volume in scoring
            confidence_threshold: Minimum confidence threshold
            lookback_days: Days to look back for scoring
        """
        self.decay_factor = decay_factor
        self.volume_weight = volume_weight
        self.confidence_threshold = confidence_threshold
        self.lookback_days = lookback_days

        # Score normalization
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def calculate_sentiment_scores(
        self, sentiment_df: pd.DataFrame, method: str = "weighted", group_by: str = "symbol"
    ) -> pd.DataFrame:
        """
        Calculate aggregated sentiment scores

        Args:
            sentiment_df: DataFrame with sentiment analysis results
            method: Scoring method ('weighted', 'exponential', 'momentum')
            group_by: Grouping column

        Returns:
            DataFrame with sentiment scores
        """
        if sentiment_df.empty:
            return pd.DataFrame()

        # Ensure required columns
        required_cols = ["sentiment_score", "sentiment_confidence", "published_date"]
        missing_cols = set(required_cols) - set(sentiment_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert dates
        sentiment_df["published_date"] = pd.to_datetime(sentiment_df["published_date"])

        # Apply scoring method
        if method == "weighted":
            scores = self._weighted_scoring(sentiment_df, group_by)
        elif method == "exponential":
            scores = self._exponential_scoring(sentiment_df, group_by)
        elif method == "momentum":
            scores = self._momentum_scoring(sentiment_df, group_by)
        else:
            raise ValueError(f"Unknown scoring method: {method}")

        return scores

    def _weighted_scoring(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """
        Calculate weighted sentiment scores

        Weights based on:
        - Recency (time decay)
        - Confidence
        - Source reliability
        """
        current_time = datetime.now()
        scores = []

        for group_value, group_df in df.groupby(group_by):
            # Calculate time weights
            time_diffs = (
                current_time - group_df["published_date"]
            ).dt.total_seconds() / 3600  # hours
            time_weights = np.exp(-time_diffs / 24) * self.decay_factor  # Daily decay

            # Confidence weights
            conf_weights = group_df["sentiment_confidence"]
            conf_weights = conf_weights.clip(lower=self.confidence_threshold)

            # Combined weights
            weights = time_weights * conf_weights
            weights = weights / weights.sum() if weights.sum() > 0 else weights

            # Calculate weighted score
            weighted_score = np.sum(group_df["sentiment_score"] * weights)

            # Calculate component scores
            components = {
                "news_sentiment": weighted_score,
                "positive_ratio": (group_df["sentiment"] == "positive").mean(),
                "negative_ratio": (group_df["sentiment"] == "negative").mean(),
                "avg_confidence": conf_weights.mean(),
            }

            # Create sentiment score
            score_obj = SentimentScore(
                symbol=group_value,
                timestamp=current_time,
                score=weighted_score,
                confidence=conf_weights.mean(),
                volume=len(group_df),
                components=components,
                metadata={"method": "weighted", "lookback_hours": time_diffs.min()},
            )

            scores.append(score_obj.to_dict())

        return pd.DataFrame(scores)

    def _exponential_scoring(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """
        Calculate exponential moving average sentiment scores
        """
        scores = []

        for group_value, group_df in df.groupby(group_by):
            # Sort by date
            group_df = group_df.sort_values("published_date")

            # Calculate EMA
            alpha = 1 - self.decay_factor
            ema_score = group_df["sentiment_score"].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

            # Calculate volatility
            score_volatility = group_df["sentiment_score"].std()

            # Components
            components = {
                "ema_sentiment": ema_score,
                "sentiment_volatility": score_volatility,
                "trend": self._calculate_trend(group_df["sentiment_score"]),
                "consistency": 1 - score_volatility if score_volatility < 1 else 0,
            }

            score_obj = SentimentScore(
                symbol=group_value,
                timestamp=datetime.now(),
                score=ema_score,
                confidence=1 - score_volatility / 2,  # Higher volatility = lower confidence
                volume=len(group_df),
                components=components,
                metadata={"method": "exponential", "alpha": alpha},
            )

            scores.append(score_obj.to_dict())

        return pd.DataFrame(scores)

    def _momentum_scoring(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """
        Calculate sentiment momentum scores
        """
        scores = []

        for group_value, group_df in df.groupby(group_by):
            # Sort by date
            group_df = group_df.sort_values("published_date")

            if len(group_df) < 2:
                continue

            # Split into recent and older
            mid_point = len(group_df) // 2
            recent_sentiment = group_df.iloc[mid_point:]["sentiment_score"].mean()
            older_sentiment = group_df.iloc[:mid_point]["sentiment_score"].mean()

            # Calculate momentum
            momentum = recent_sentiment - older_sentiment

            # Calculate acceleration
            if len(group_df) >= 4:
                quarter = len(group_df) // 4
                very_recent = group_df.iloc[-quarter:]["sentiment_score"].mean()
                acceleration = very_recent - recent_sentiment
            else:
                acceleration = 0

            # Components
            components = {
                "current_sentiment": recent_sentiment,
                "momentum": momentum,
                "acceleration": acceleration,
                "momentum_strength": abs(momentum),
            }

            # Combined score (current + momentum effect)
            combined_score = recent_sentiment + momentum * 0.3
            combined_score = np.clip(combined_score, -1, 1)

            score_obj = SentimentScore(
                symbol=group_value,
                timestamp=datetime.now(),
                score=combined_score,
                confidence=self._calculate_momentum_confidence(momentum, acceleration),
                volume=len(group_df),
                components=components,
                metadata={
                    "method": "momentum",
                    "time_span_hours": (
                        group_df["published_date"].max() - group_df["published_date"].min()
                    ).total_seconds()
                    / 3600,
                },
            )

            scores.append(score_obj.to_dict())

        return pd.DataFrame(scores)

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend coefficient"""
        if len(series) < 2:
            return 0

        # Simple linear regression slope
        x = np.arange(len(series))
        y = series.values

        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0

        x = x[mask]
        y = y[mask]

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        return float(slope)

    def _calculate_momentum_confidence(self, momentum: float, acceleration: float) -> float:
        """Calculate confidence based on momentum consistency"""
        # Higher confidence if momentum and acceleration align
        if np.sign(momentum) == np.sign(acceleration):
            confidence = min(abs(momentum) + abs(acceleration) / 2, 1.0)
        else:
            confidence = max(0.3, 1 - abs(acceleration))

        return confidence

    def create_composite_score(
        self, scores: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create composite sentiment score from multiple methods

        Args:
            scores: Dictionary of scores by method
            weights: Weights for each method

        Returns:
            DataFrame with composite scores
        """
        if not scores:
            return pd.DataFrame()

        # Default weights
        if weights is None:
            weights = {"weighted": 0.4, "exponential": 0.3, "momentum": 0.3}

        # Combine scores
        composite_scores = []

        # Get all symbols
        all_symbols = set()
        for method_scores in scores.values():
            if "symbol" in method_scores.columns:
                all_symbols.update(method_scores["symbol"].unique())

        for symbol in all_symbols:
            symbol_scores = {}
            total_weight = 0

            # Collect scores from each method
            for method, method_df in scores.items():
                if method not in weights:
                    continue

                symbol_data = method_df[method_df["symbol"] == symbol]
                if not symbol_data.empty:
                    symbol_scores[method] = symbol_data.iloc[0]["score"]
                    total_weight += weights[method]

            if symbol_scores and total_weight > 0:
                # Calculate weighted average
                composite_score = sum(
                    score * weights[method] / total_weight
                    for method, score in symbol_scores.items()
                )

                # Calculate confidence as weighted average
                composite_confidence = np.mean(
                    [
                        method_df[method_df["symbol"] == symbol].iloc[0]["confidence"]
                        for method, method_df in scores.items()
                        if not method_df[method_df["symbol"] == symbol].empty
                    ]
                )

                composite_scores.append(
                    {
                        "symbol": symbol,
                        "composite_score": composite_score,
                        "composite_confidence": composite_confidence,
                        "component_scores": symbol_scores,
                        "timestamp": datetime.now(),
                    }
                )

        return pd.DataFrame(composite_scores)

    def get_sentiment_signals(
        self, scores_df: pd.DataFrame, threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate trading signals from sentiment scores

        Args:
            scores_df: DataFrame with sentiment scores
            threshold: Signal threshold

        Returns:
            DataFrame with trading signals
        """
        []

        for _, row in scores_df.iterrows():
            score = row.get("composite_score", row.get("score", 0))
            confidence = row.get("composite_confidence", row.get("confidence", 0))

            # Generate signal
            if score > threshold and confidence > self.confidence_threshold:
                signal = "STRONG_POSITIVE" if score > threshold * 2 else "POSITIVE"
            elif score < -threshold and confidence > self.confidence_threshold:
                signal = "STRONG_NEGATIVE" if score < -threshold * 2 else "NEGATIVE"
            else:
                signal = "NEUTRAL"

            signals.append(
                {
                    "symbol": row["symbol"],
                    "signal": signal,
                    "strength": abs(score),
                    "confidence": confidence,
                    "timestamp": row.get("timestamp", datetime.now()),
                }
            )

        return pd.DataFrame(signals)

    def calculate_market_sentiment(
        self, sentiment_df: pd.DataFrame, sector_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate overall market sentiment

        Args:
            sentiment_df: DataFrame with sentiment data
            sector_map: Symbol to sector mapping

        Returns:
            Market sentiment metrics
        """
        # Overall market sentiment
        market_score = sentiment_df["sentiment_score"].mean()
        market_volatility = sentiment_df["sentiment_score"].std()

        # Breadth indicators
        positive_ratio = (sentiment_df["sentiment_score"] > 0).mean()
        strong_positive = (sentiment_df["sentiment_score"] > 0.5).mean()
        strong_negative = (sentiment_df["sentiment_score"] < -0.5).mean()

        market_sentiment = {
            "market_score": market_score,
            "market_volatility": market_volatility,
            "positive_breadth": positive_ratio,
            "strong_positive_ratio": strong_positive,
            "strong_negative_ratio": strong_negative,
            "sentiment_dispersion": sentiment_df["sentiment_score"].mad(),
            "timestamp": datetime.now(),
        }

        # Sector sentiment if mapping provided
        if sector_map and "symbol" in sentiment_df.columns:
            sentiment_df["sector"] = sentiment_df["symbol"].map(sector_map)

            sector_sentiment = {}
            for sector, sector_df in sentiment_df.groupby("sector"):
                sector_sentiment[sector] = {
                    "score": sector_df["sentiment_score"].mean(),
                    "volume": len(sector_df),
                    "volatility": sector_df["sentiment_score"].std(),
                }

            market_sentiment["sector_sentiment"] = sector_sentiment

        return market_sentiment

    def save_scores(self, scores: pd.DataFrame, filepath: str) -> None:
        """Save sentiment scores to file"""
        scores.to_csv(filepath, index=False)
        logger.info(f"Saved sentiment scores to {filepath}")

    def load_scores(self, filepath: str) -> pd.DataFrame:
        """Load sentiment scores from file"""
        return pd.read_csv(filepath, parse_dates=["timestamp"])
