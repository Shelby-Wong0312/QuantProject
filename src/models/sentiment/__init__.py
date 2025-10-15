"""
Sentiment Analysis Module for Financial News

This module provides:
- FinBERT-based sentiment analysis
- News data collection and processing
- Real-time sentiment monitoring
- Integration with trading features
"""

from .finbert_analyzer import FinBERTAnalyzer
from .news_collector import NewsCollector
from .sentiment_scorer import SentimentScorer
from .data_pipeline import SentimentDataPipeline

__all__ = ["FinBERTAnalyzer", "NewsCollector", "SentimentScorer", "SentimentDataPipeline"]
