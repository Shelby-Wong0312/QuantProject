"""
News data collection module
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import requests
from bs4 import BeautifulSoup
import feedparser
import asyncio
import aiohttp
from pathlib import Path
import json
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data structure"""

    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    symbols: List[str]
    category: Optional[str] = None
    author: Optional[str] = None
    sentiment_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "published_date": self.published_date.isoformat(),
            "symbols": self.symbols,
            "category": self.category,
            "author": self.author,
            "sentiment_score": self.sentiment_score,
            "metadata": self.metadata or {},
        }


class NewsSource(ABC):
    """Abstract base class for news sources"""

    @abstractmethod
    async def fetch_articles(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[NewsArticle]:
        """Fetch articles for given symbols and date range"""
        pass


class RSSNewsSource(NewsSource):
    """RSS feed news source"""

    def __init__(self, feed_urls: Dict[str, str]):
        """
        Initialize RSS news source

        Args:
            feed_urls: Dictionary mapping source names to RSS URLs
        """
        self.feed_urls = feed_urls

    async def fetch_articles(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[NewsArticle]:
        """Fetch articles from RSS feeds"""
        articles = []

        for source_name, feed_url in self.feed_urls.items():
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    # Parse published date
                    published = None
                    if hasattr(entry, "published_parsed"):
                        published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    elif hasattr(entry, "updated_parsed"):
                        published = datetime.fromtimestamp(time.mktime(entry.updated_parsed))

                    if not published:
                        continue

                    # Check date range
                    if published < start_date or published > end_date:
                        continue

                    # Extract content
                    content = entry.get("summary", "")
                    if hasattr(entry, "content"):
                        content = entry.content[0].value

                    # Check if any symbol is mentioned
                    mentioned_symbols = []
                    full_text = f"{entry.title} {content}".lower()

                    for symbol in symbols:
                        if symbol.lower() in full_text:
                            mentioned_symbols.append(symbol)

                    if mentioned_symbols:
                        article = NewsArticle(
                            title=entry.title,
                            content=content,
                            source=source_name,
                            url=entry.link,
                            published_date=published,
                            mentioned_symbols,
                            author=entry.get("author"),
                            metadata={"feed_source": feed_url},
                        )
                        articles.append(article)

            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {str(e)}")

        return articles


class FinancialNewsAPI(NewsSource):
    """Financial news API wrapper (mock implementation)"""

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize financial news API

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = None

    async def fetch_articles(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> List[NewsArticle]:
        """Fetch articles from financial news API"""
        articles = []

        # This is a mock implementation
        # In production, you would integrate with actual news APIs like:
        # - Alpha Vantage News
        # - Bloomberg API
        # - Reuters API
        # - NewsAPI.org

        # Mock data for demonstration
        mock_articles = [
            {
                "title": f"{symbol} Reports Strong Quarterly Earnings",
                "content": f"Company {symbol} exceeded analyst expectations with robust revenue growth.",
                "symbol": symbol,
                "sentiment": 0.8,
            }
            for symbol in symbols[:3]  # Limit for demo
        ]

        for article_data in mock_articles:
            article = NewsArticle(
                title=article_data["title"],
                content=article_data["content"],
                source="Financial News API",
                url=f"https://example.com/news/{article_data['symbol']}",
                published_date=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                [article_data["symbol"]],
                sentiment_score=article_data["sentiment"],
            )
            articles.append(article)

        return articles


class NewsCollector:
    """Main news collection orchestrator"""

    def __init__(
        self,
        sources: Optional[List[NewsSource]] = None,
        cache_dir: Optional[Union[str, Path]] = "./news_cache",
        cache_expiry_hours: int = 24,
    ):
        """
        Initialize news collector

        Args:
            sources: List of news sources
            cache_dir: Directory for caching news
            cache_expiry_hours: Cache expiry time in hours
        """
        self.sources = sources or self._get_default_sources()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours

    def _get_default_sources(self) -> List[NewsSource]:
        """Get default news sources"""
        # Default RSS feeds for financial news
        rss_feeds = {
            "Reuters Finance": "https://feeds.reuters.com/reuters/businessNews",
            "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
            "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories",
            "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        }

        sources = [
            RSSNewsSource(rss_feeds),
            # Add more sources as needed
        ]

        # Add API sources if credentials available
        if self._has_api_credentials():
            sources.append(
                FinancialNewsAPI(api_key="your_api_key", base_url="https://api.example.com")
            )

        return sources

    def _has_api_credentials(self) -> bool:
        """Check if API credentials are available"""
        # Check environment variables or config file
        import os

        return bool(os.getenv("NEWS_API_KEY"))

    async def collect_news(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Collect news articles for given symbols

        Args:
            symbols: List of stock symbols
            start_date: Start date for news collection
            end_date: End date for news collection
            use_cache: Whether to use cached news

        Returns:
            DataFrame with news articles
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()

        # Check cache
        if use_cache:
            cached_articles = self._load_from_cache(symbols, start_date, end_date)
            if cached_articles is not None:
                logger.info(f"Loaded {len(cached_articles)} articles from cache")
                return cached_articles

        # Collect from all sources
        all_articles = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.sources:
                task = source.fetch_articles(symbols, start_date, end_date)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error collecting news: {str(result)}")
                else:
                    all_articles.extend(result)

        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame([article.to_dict() for article in all_articles])
            df["published_date"] = pd.to_datetime(df["published_date"])
            df = df.sort_values("published_date", ascending=False)

            # Remove duplicates
            df = df.drop_duplicates(subset=["title", "source"])

            # Save to cache
            if use_cache:
                self._save_to_cache(df, symbols)

            logger.info(f"Collected {len(df)} unique articles")
            return df
        else:
            logger.warning("No articles collected")
            return pd.DataFrame()

    def collect_news_sync(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Synchronous wrapper for collect_news"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.collect_news(symbols, start_date, end_date, use_cache))

    def _load_from_cache(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load news from cache"""
        cache_file = self._get_cache_filename(symbols)

        if not cache_file.exists():
            return None

        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > timedelta(hours=self.cache_expiry_hours):
            logger.info("Cache expired")
            return None

        try:
            df = pd.read_csv(cache_file, parse_dates=["published_date"])

            # Filter by date range
            df = df[(df["published_date"] >= start_date) & (df["published_date"] <= end_date)]

            return df
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, symbols: List[str]) -> None:
        """Save news to cache"""
        cache_file = self._get_cache_filename(symbols)

        try:
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved {len(df)} articles to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _get_cache_filename(self, symbols: List[str]) -> Path:
        """Get cache filename for symbols"""
        symbols_str = "_".join(sorted(symbols[:5]))  # Limit to 5 symbols
        return self.cache_dir / f"news_{symbols_str}.csv"

    def get_latest_news(self, symbols: List[str], hours: int = 24) -> pd.DataFrame:
        """Get latest news for symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)

        return self.collect_news_sync(symbols, start_date, end_date)

    def stream_news(
        self, symbols: List[str], callback: callable, interval_seconds: int = 300
    ) -> None:
        """
        Stream news updates

        Args:
            symbols: List of symbols to monitor
            callback: Function to call with new articles
            interval_seconds: Update interval
        """
        logger.info(f"Starting news stream for {symbols}")

        last_update = datetime.now()

        while True:
            try:
                # Get recent news
                news_df = self.get_latest_news(symbols, hours=1)

                # Filter new articles
                if not news_df.empty:
                    new_articles = news_df[news_df["published_date"] > last_update]

                    if not new_articles.empty:
                        logger.info(f"Found {len(new_articles)} new articles")
                        callback(new_articles)

                last_update = datetime.now()

            except Exception as e:
                logger.error(f"Error in news stream: {str(e)}")

            time.sleep(interval_seconds)
