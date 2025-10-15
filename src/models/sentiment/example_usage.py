"""
Example usage of sentiment analysis module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio

from news_collector import NewsCollector, NewsArticle
from finbert_analyzer import FinBERTAnalyzer
from sentiment_scorer import SentimentScorer
from data_pipeline import SentimentDataPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_news_collection():
    """Example of collecting financial news"""
    print("\n" + "=" * 60)
    print("NEWS COLLECTION EXAMPLE")
    print("=" * 60)

    # Initialize news collector
    collector = NewsCollector()

    # Collect news for specific symbols
    ["AAPL", "GOOGL", "MSFT"]
    news_df = collector.collect_news_sync(
        symbols, start_date=datetime.now() - timedelta(days=3), end_date=datetime.now()
    )

    if not news_df.empty:
        print(f"\nCollected {len(news_df)} articles")
        print(f"Sources: {news_df['source'].unique()}")
        print("\nSample articles:")
        for idx, article in news_df.head(3).iterrows():
            print(f"\n- {article['title'][:80]}...")
            print(f"  Source: {article['source']}")
            print(f"  Date: {article['published_date']}")
            print(f"  Symbols: {article['symbols']}")
    else:
        print("No news articles found")


def example_sentiment_analysis():
    """Example of analyzing sentiment with FinBERT"""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS EXAMPLE")
    print("=" * 60)

    # Initialize analyzer
    analyzer = FinBERTAnalyzer()

    # Sample financial texts
    texts = [
        "Apple reported record-breaking quarterly earnings, exceeding analyst expectations by 15%.",
        "The Federal Reserve's hawkish stance on interest rates sent markets tumbling today.",
        "Tesla's production numbers came in line with expectations for the quarter.",
        "Banking sector faces headwinds as credit losses mount amid economic uncertainty.",
        "Microsoft announces strategic AI partnership, boosting investor confidence.",
    ]

    # Analyze sentiment
    results = analyzer.analyze_sentiment(texts, return_all_scores=True)

    print("\nSentiment Analysis Results:")
    print("-" * 40)

    for result in results:
        print(f"\nText: {result['text'][:60]}...")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Score: {result['sentiment_score']:.3f}")
        print(
            f"Probabilities: Pos={result['scores']['positive']:.3f}, "
            f"Neu={result['scores']['neutral']:.3f}, "
            f"Neg={result['scores']['negative']:.3f}"
        )


def example_sentiment_scoring():
    """Example of sentiment scoring system"""
    print("\n" + "=" * 60)
    print("SENTIMENT SCORING EXAMPLE")
    print("=" * 60)

    # Create sample sentiment data
    sentiment_data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT", "MSFT"],
            "sentiment_score": [0.8, 0.6, -0.2, 0.3, 0.5, -0.4, -0.6],
            "sentiment_confidence": [0.9, 0.8, 0.7, 0.85, 0.9, 0.75, 0.8],
            "sentiment": [
                "positive",
                "positive",
                "negative",
                "positive",
                "positive",
                "negative",
                "negative",
            ],
            "published_date": [datetime.now() - timedelta(hours=i) for i in [1, 3, 6, 2, 5, 1, 4]],
        }
    )

    # Initialize scorer
    scorer = SentimentScorer()

    # Calculate scores using different methods
    print("\n1. Weighted Scoring:")
    weighted_scores = scorer.calculate_sentiment_scores(sentiment_data, method="weighted")
    print(weighted_scores[["symbol", "score", "confidence", "volume"]])

    print("\n2. Exponential Scoring:")
    exp_scores = scorer.calculate_sentiment_scores(sentiment_data, method="exponential")
    print(exp_scores[["symbol", "score", "confidence"]])

    print("\n3. Momentum Scoring:")
    momentum_scores = scorer.calculate_sentiment_scores(sentiment_data, method="momentum")
    print(momentum_scores[["symbol", "score", "confidence"]])

    # Create composite score
    print("\n4. Composite Scores:")
    composite = scorer.create_composite_score(
        {"weighted": weighted_scores, "exponential": exp_scores, "momentum": momentum_scores}
    )
    print(composite)

    # Generate trading signals
    print("\n5. Trading Signals:")
    scorer.get_sentiment_signals(composite)
    print(signals)


def example_complete_pipeline():
    """Example of complete sentiment analysis pipeline"""
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE EXAMPLE")
    print("=" * 60)

    # Initialize pipeline
    pipeline = SentimentDataPipeline()

    # Process symbols
    ["AAPL", "MSFT", "GOOGL"]

    print(f"\nProcessing sentiment for {symbols}...")
    results = pipeline.process_symbols(symbols=symbols, news_lookback_days=3)

    # Check results
    if results:
        # Generate report
        pipeline.generate_sentiment_report(results, output_file="sentiment_report.txt")
        print("\nGenerated Report:")
        print(report)

        # Get composite scores
        if "composite_scores" in results:
            print("\n\nComposite Sentiment Scores:")
            print(results["composite_scores"])
    else:
        print("No results generated")


def example_live_monitoring():
    """Example of live sentiment monitoring"""
    print("\n" + "=" * 60)
    print("LIVE MONITORING EXAMPLE")
    print("=" * 60)

    # Initialize pipeline
    pipeline = SentimentDataPipeline()

    # Define callback function
    def on_sentiment_update(results):
        """Callback for sentiment updates"""
        print(f"\n[{datetime.now()}] Sentiment Update:")

        if "composite_scores" in results:
            scores = results["composite_scores"]
            for _, row in scores.iterrows():
                print(
                    f"  {row['symbol']}: {row['composite_score']:.3f} "
                    f"(confidence: {row['composite_confidence']:.3f})"
                )

        if "signals" in results:
            results["signals"]
            strong_signals = signals[signals["signal"].str.contains("STRONG")]
            if not strong_signals.empty:
                print("\n  Strong Signals:")
                for _, signal in strong_signals.iterrows():
                    print(f"    {signal['symbol']}: {signal['signal']}")

    # Note: This would run continuously in production
    print("\nStarting live monitoring (demo - will run once)...")

    # For demo, just process once
    ["AAPL", "MSFT"]
    results = pipeline.process_symbols(symbols, news_lookback_days=1)
    on_sentiment_update(results)


def example_ml_features():
    """Example of creating sentiment features for ML"""
    print("\n" + "=" * 60)
    print("ML FEATURE CREATION EXAMPLE")
    print("=" * 60)

    # Create sample market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
    market_data = pd.DataFrame(
        {
            "datetime": dates,
            "AAPL_close": 150 + np.random.randn(100) * 2,
            "MSFT_close": 300 + np.random.randn(100) * 3,
            "volume": np.random.randint(1000000, 5000000, 100),
        }
    )
    market_data.set_index("datetime", inplace=True)

    # Initialize pipeline
    pipeline = SentimentDataPipeline()

    # Create sentiment features
    ["AAPL", "MSFT"]
    enhanced_data = pipeline.create_sentiment_features(
        market_data, symbols=symbols, lookback_days=3
    )

    # Show new features
    sentiment_cols = [col for col in enhanced_data.columns if "sentiment" in col]

    print(f"\nAdded {len(sentiment_cols)} sentiment features:")
    for col in sentiment_cols:
        print(f"  - {col}")

    if sentiment_cols:
        print("\nSample data with sentiment features:")
        print(enhanced_data[sentiment_cols].tail())


def example_custom_news_source():
    """Example of adding custom news source"""
    print("\n" + "=" * 60)
    print("CUSTOM NEWS SOURCE EXAMPLE")
    print("=" * 60)

    # Create custom RSS feeds
    from news_collector import RSSNewsSource

    custom_feeds = {
        "Custom Finance": "https://example.com/finance.rss",
        "Market News": "https://example.com/markets.rss",
    }

    # Create collector with custom source
    custom_source = RSSNewsSource(custom_feeds)
    collector = NewsCollector(sources=[custom_source])

    print("Configured custom news sources:")
    for name, url in custom_feeds.items():
        print(f"  - {name}: {url}")

    # In production, you would collect news:
    # news_df = collector.collect_news_sync(['AAPL'], use_cache=False)


def example_fine_tuning():
    """Example of fine-tuning FinBERT"""
    print("\n" + "=" * 60)
    print("FINBERT FINE-TUNING EXAMPLE")
    print("=" * 60)

    # Create training data (in practice, use real labeled data)
    train_texts = [
        "Company reported strong earnings growth",
        "Bankruptcy filing sends shares plummeting",
        "Quarterly results met analyst expectations",
        "SEC investigation announced into accounting practices",
        "New product launch drives revenue higher",
    ]

    train_labels = ["positive", "negative", "neutral", "negative", "positive"]

    # For demo, we'll just show the setup
    print("Fine-tuning setup:")
    print(f"Training samples: {len(train_texts)}")
    print(f"Labels: {set(train_labels)}")

    # In practice, you would fine-tune:
    # analyzer = FinBERTAnalyzer()
    # results = analyzer.fine_tune(
    #     train_texts, train_labels,
    #     num_epochs=3,
    #     output_dir='./finbert_custom'
    # )


if __name__ == "__main__":
    # Run examples
    print("SENTIMENT ANALYSIS MODULE EXAMPLES")
    print("=" * 60)

    # Note: Some examples require API keys or internet connection
    try:
        example_sentiment_analysis()  # This should work offline
        example_sentiment_scoring()  # This should work offline
        example_ml_features()  # This should work offline

        # These require internet/API access:
        # example_news_collection()
        # example_complete_pipeline()
        # example_live_monitoring()

        example_custom_news_source()  # Demo only
        example_fine_tuning()  # Demo only

    except Exception as e:
        print(f"\nError in examples: {str(e)}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
