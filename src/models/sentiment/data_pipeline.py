"""
Data pipeline for sentiment analysis workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import json
import schedule
import time
from concurrent.futures import ThreadPoolExecutor

from .news_collector import NewsCollector, NewsArticle
from .finbert_analyzer import FinBERTAnalyzer
from .sentiment_scorer import SentimentScorer


logger = logging.getLogger(__name__)


class SentimentDataPipeline:
    """
    End-to-end pipeline for financial sentiment analysis
    """

    def __init__(
        self,
        news_collector: Optional[NewsCollector] = None,
        analyzer: Optional[FinBERTAnalyzer] = None,
        scorer: Optional[SentimentScorer] = None,
        output_dir: Union[str, Path] = "./sentiment_output",
        cache_results: bool = True,
    ):
        """
        Initialize sentiment data pipeline

        Args:
            news_collector: News collection instance
            analyzer: FinBERT analyzer instance
            scorer: Sentiment scorer instance
            output_dir: Directory for output files
            cache_results: Whether to cache intermediate results
        """
        self.news_collector = news_collector or NewsCollector()
        self.analyzer = analyzer or FinBERTAnalyzer()
        self.scorer = scorer or SentimentScorer()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_results = cache_results
        self._cache = {}

        # Pipeline state
        self.is_running = False
        self.last_update = None

    def process_symbols(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        news_lookback_days: int = 7,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process sentiment analysis for given symbols

        Args:
            symbols: List of stock symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            news_lookback_days: Days to look back for news

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Processing sentiment for {len(symbols)} symbols")

        # Set date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=news_lookback_days)

        results = {}

        # Step 1: Collect news
        logger.info("Step 1: Collecting news articles")
        news_df = self.news_collector.collect_news_sync(
            symbols, start_date=start_date, end_date=end_date, use_cache=self.cache_results
        )

        if news_df.empty:
            logger.warning("No news articles found")
            return results

        results["raw_news"] = news_df
        logger.info(f"Collected {len(news_df)} articles")

        # Step 2: Analyze sentiment
        logger.info("Step 2: Analyzing sentiment with FinBERT")
        sentiment_df = self.analyzer.analyze_news_sentiment(news_df)
        results["sentiment_analysis"] = sentiment_df

        # Step 3: Calculate scores
        logger.info("Step 3: Calculating sentiment scores")

        # Calculate scores using different methods
        weighted_scores = self.scorer.calculate_sentiment_scores(sentiment_df, method="weighted")
        exponential_scores = self.scorer.calculate_sentiment_scores(
            sentiment_df, method="exponential"
        )
        momentum_scores = self.scorer.calculate_sentiment_scores(sentiment_df, method="momentum")

        # Create composite scores
        composite_scores = self.scorer.create_composite_score(
            {
                "weighted": weighted_scores,
                "exponential": exponential_scores,
                "momentum": momentum_scores,
            }
        )

        results["weighted_scores"] = weighted_scores
        results["exponential_scores"] = exponential_scores
        results["momentum_scores"] = momentum_scores
        results["composite_scores"] = composite_scores

        # Step 4: Generate signals
        logger.info("Step 4: Generating trading signals")
        self.scorer.get_sentiment_signals(composite_scores)
        results["signals"] = signals

        # Step 5: Calculate market sentiment
        market_sentiment = self.scorer.calculate_market_sentiment(sentiment_df)
        results["market_sentiment"] = market_sentiment

        # Save results if caching enabled
        if self.cache_results:
            self._save_results(results)

        self.last_update = datetime.now()

        return results

    def get_live_sentiment(self, symbols: List[str], lookback_hours: int = 24) -> pd.DataFrame:
        """
        Get live sentiment scores for symbols

        Args:
            symbols: List of symbols
            lookback_hours: Hours to look back

        Returns:
            DataFrame with current sentiment scores
        """
        # Get recent news
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)

        results = self.process_symbols(symbols=symbols, start_date=start_date, end_date=end_date)

        if "composite_scores" in results:
            return results["composite_scores"]
        else:
            return pd.DataFrame()

    def create_sentiment_features(
        self, market_data: pd.DataFrame, symbols: List[str], lookback_days: int = 7
    ) -> pd.DataFrame:
        """
        Create sentiment features for ML models

        Args:
            market_data: Market data DataFrame
            symbols: List of symbols
            lookback_days: Days to look back for sentiment

        Returns:
            DataFrame with sentiment features added
        """
        # Get sentiment data
        results = self.process_symbols(symbols=symbols, news_lookback_days=lookback_days)

        if not results or "composite_scores" not in results:
            logger.warning("No sentiment data available")
            return market_data

        # Add sentiment features
        enhanced_data = market_data.copy()
        scores_df = results["composite_scores"]

        for symbol in symbols:
            if symbol in scores_df["symbol"].values:
                symbol_scores = scores_df[scores_df["symbol"] == symbol].iloc[0]

                # Add features
                enhanced_data[f"{symbol}_sentiment_score"] = symbol_scores["composite_score"]
                enhanced_data[f"{symbol}_sentiment_confidence"] = symbol_scores[
                    "composite_confidence"
                ]

                # Add component scores if available
                if "component_scores" in symbol_scores:
                    for method, score in symbol_scores["component_scores"].items():
                        enhanced_data[f"{symbol}_sentiment_{method}"] = score

        # Add market sentiment features
        if "market_sentiment" in results:
            market_sent = results["market_sentiment"]
            enhanced_data["market_sentiment_score"] = market_sent["market_score"]
            enhanced_data["market_sentiment_volatility"] = market_sent["market_volatility"]
            enhanced_data["market_positive_breadth"] = market_sent["positive_breadth"]

        return enhanced_data

    def run_continuous_monitoring(
        self,
        symbols: List[str],
        update_interval_minutes: int = 60,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Run continuous sentiment monitoring

        Args:
            symbols: Symbols to monitor
            update_interval_minutes: Update interval
            callback: Callback function for updates
        """
        logger.info(f"Starting continuous monitoring for {symbols}")
        self.is_running = True

        def update_sentiment():
            """Update sentiment data"""
            try:
                results = self.process_symbols(symbols=symbols, news_lookback_days=1)

                # Call callback if provided
                if callback and results:
                    callback(results)

                logger.info(f"Updated sentiment at {datetime.now()}")

            except Exception as e:
                logger.error(f"Error in sentiment update: {str(e)}")

        # Schedule updates
        schedule.every(update_interval_minutes).minutes.do(update_sentiment)

        # Run initial update
        update_sentiment()

        # Run scheduler
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("Stopped sentiment monitoring")

    def generate_sentiment_report(
        self, results: Dict[str, Any], output_file: Optional[str] = None
    ) -> str:
        """
        Generate sentiment analysis report

        Args:
            results: Analysis results
            output_file: Optional output file path

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SENTIMENT ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)

        # News summary
        if "raw_news" in results:
            news_df = results["raw_news"]
            report_lines.append("\nNEWS SUMMARY:")
            report_lines.append(f"Total articles: {len(news_df)}")
            report_lines.append(
                f"Date range: {news_df['published_date'].min()} to {news_df['published_date'].max()}"
            )
            report_lines.append(f"Unique sources: {news_df['source'].nunique()}")

        # Sentiment scores
        if "composite_scores" in results:
            scores_df = results["composite_scores"]
            report_lines.append("\nSENTIMENT SCORES:")

            for _, row in scores_df.iterrows():
                report_lines.append(f"\n{row['symbol']}:")
                report_lines.append(f"  Composite Score: {row['composite_score']:.3f}")
                report_lines.append(f"  Confidence: {row['composite_confidence']:.3f}")

                if "component_scores" in row:
                    report_lines.append("  Component Scores:")
                    for method, score in row["component_scores"].items():
                        report_lines.append(f"    - {method}: {score:.3f}")

        # Trading signals
        if "signals" in results:
            signals_df = results["signals"]
            report_lines.append("\nTRADING SIGNALS:")

            for _, signal in signals_df.iterrows():
                report_lines.append(
                    f"{signal['symbol']}: {signal['signal']} "
                    f"(strength: {signal['strength']:.3f}, "
                    f"confidence: {signal['confidence']:.3f})"
                )

        # Market sentiment
        if "market_sentiment" in results:
            market = results["market_sentiment"]
            report_lines.append("\nMARKET SENTIMENT:")
            report_lines.append(f"  Overall Score: {market['market_score']:.3f}")
            report_lines.append(f"  Volatility: {market['market_volatility']:.3f}")
            report_lines.append(f"  Positive Breadth: {market['positive_breadth']:.2%}")

            if "sector_sentiment" in market:
                report_lines.append("\n  Sector Sentiment:")
                for sector, sent in market["sector_sentiment"].items():
                    report_lines.append(f"    {sector}: {sent['score']:.3f}")

        report_lines.append("\n" + "=" * 60)

        report_text = "\n".join(report_lines)

        # Save report if requested
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for key, data in results.items():
            if isinstance(data, pd.DataFrame):
                filename = self.output_dir / f"{key}_{timestamp}.csv"
                data.to_csv(filename, index=False)
            elif isinstance(data, dict):
                filename = self.output_dir / f"{key}_{timestamp}.json"
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2, default=str)

    def load_historical_sentiment(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Load historical sentiment data

        Args:
            symbol: Stock symbol
            days: Number of days to load

        Returns:
            DataFrame with historical sentiment
        """
        # Find relevant files
        pattern = "composite_scores_*.csv"
        files = list(self.output_dir.glob(pattern))

        if not files:
            logger.warning("No historical sentiment data found")
            return pd.DataFrame()

        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Load and combine data
        all_data = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for file in files[:days]:  # Limit files to check
            try:
                df = pd.read_csv(file)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Filter by date and symbol
                    df = df[(df["timestamp"] >= cutoff_date) & (df["symbol"] == symbol)]

                    if not df.empty:
                        all_data.append(df)

            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")

        if all_data:
            return pd.concat(all_data, ignore_index=True).sort_values("timestamp")
        else:
            return pd.DataFrame()
