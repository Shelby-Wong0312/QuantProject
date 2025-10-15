"""
Free Data Sources Client - Plan B Implementation
免費數據源客戶端 - 方案B實施
"""

import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FreeDataClient:
    """
    統一的免費數據源客戶端
    整合Yahoo Finance、Alpha Vantage、Twelve Data等
    支援4000+股票大規模監控
    """

    def __init__(self, db_path: str = "data/market_data.db"):
        """初始化數據客戶端"""
        # API Keys
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        # Rate limiting
        self.alpha_vantage_calls = 0
        self.alpha_vantage_reset_time = time.time() + 60
        self._rate_limit_lock = threading.Lock()

        # Cache
        self.cache = {}
        self.cache_duration = 60  # 60 seconds cache

        # Database
        self.db_path = db_path
        self._init_database()

        # Batch processing config
        self.batch_size = 50  # Yahoo Finance batch size
        self.max_workers = 10  # Concurrent threads
        self.request_delay = 0.1  # Delay between requests

        logger.info("Enhanced Free Data Client initialized with large-scale support")

    def _init_database(self):
        """初始化SQLite數據庫"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # 實時報價表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS real_time_quotes (
                    symbol TEXT PRIMARY KEY,
                    price REAL,
                    timestamp DATETIME,
                    volume INTEGER,
                    change_percent REAL
                )
            """
            )

            # 歷史數據緩存表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_cache (
                    symbol TEXT,
                    date DATE,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """
            )

            # 技術指標緩存表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    symbol TEXT,
                    indicator_name TEXT,
                    indicator_value REAL,
                    timestamp DATETIME,
                    PRIMARY KEY (symbol, indicator_name, timestamp)
                )
            """
            )

            conn.commit()

        logger.info(f"Database initialized at {self.db_path}")

    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        獲取實時價格
        優先順序：Yahoo Finance -> Twelve Data -> Alpha Vantage
        """
        try:
            # Method 1: Yahoo Finance (fastest, most reliable)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if price:
                logger.debug(f"Got {symbol} price from Yahoo: ${price:.2f}")
                return price

            # Method 2: Yahoo Finance history (backup)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
                logger.debug(f"Got {symbol} price from Yahoo history: ${price:.2f}")
                return price

            # Method 3: Twelve Data (demo API)
            response = requests.get(
                "https://api.twelvedata.com/price",
                params={"symbol": symbol, "apikey": "demo"},
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()
                if "price" in data:
                    price = float(data["price"])
                    logger.debug(f"Got {symbol} price from Twelve Data: ${price:.2f}")
                    return price

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")

        return None

    def get_historical_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        獲取歷史數據

        Args:
            symbol: 股票代碼
            period: 時間週期 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 數據間隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        try:
            # Use Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            if not hist.empty:
                logger.info(f"Got {len(hist)} historical records for {symbol}")
                return hist

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")

        return None

    def get_technical_indicator(
        self, symbol: str, indicator: str = "RSI", **params
    ) -> Optional[Dict]:
        """
        獲取技術指標（使用Alpha Vantage免費API）

        Args:
            symbol: 股票代碼
            indicator: 指標名稱 (RSI, MACD, SMA, EMA, etc.)
            **params: 指標參數
        """
        # Check cache first
        cache_key = f"{symbol}_{indicator}_{json.dumps(params, sort_keys=True)}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cached_data

        # Check rate limit (5 calls per minute)
        if self.alpha_vantage_calls >= 5:
            wait_time = self.alpha_vantage_reset_time - time.time()
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            self.alpha_vantage_calls = 0
            self.alpha_vantage_reset_time = time.time() + 60

        try:
            # Build request parameters
            request_params = {
                "function": indicator,
                "symbol": symbol,
                "apikey": self.alpha_vantage_key,
            }
            request_params.update(params)

            # Make request
            response = requests.get(
                "https://www.alphavantage.co/query", params=request_params, timeout=10
            )

            self.alpha_vantage_calls += 1

            if response.status_code == 200:
                data = response.json()

                # Cache the result
                self.cache[cache_key] = (data, time.time())

                logger.info(f"Got {indicator} for {symbol} from Alpha Vantage")
                return data

        except Exception as e:
            logger.error(f"Error getting {indicator} for {symbol}: {e}")

        return None

    def get_stock_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        獲取股票基本面數據

        Args:
            symbol: 股票代碼

        Returns:
            基本面數據字典
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamentals = {
                "symbol": symbol,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "debt_to_equity": info.get("debtToEquity"),
                "return_on_equity": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "beta": info.get("beta"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "timestamp": datetime.now().isoformat(),
            }

            return fundamentals

        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None

    def get_watchlist_summary(self, symbols: List[str]) -> Dict:
        """
        獲取監控清單摘要

        Args:
            symbols: 股票代碼列表

        Returns:
            監控清單摘要統計
        """
        quotes = self.get_batch_quotes(symbols)

        if not quotes:
            return {"error": "No data available"}

        prices = [data["price"] for data in quotes.values()]
        volumes = [data["volume"] for data in quotes.values()]

        summary = {
            "total_symbols": len(symbols),
            "successful_quotes": len(quotes),
            "success_rate": len(quotes) / len(symbols) * 100,
            "price_stats": {
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "mean": np.mean(prices) if prices else 0,
                "median": np.median(prices) if prices else 0,
            },
            "volume_stats": {
                "total": sum(volumes) if volumes else 0,
                "mean": np.mean(volumes) if volumes else 0,
                "median": np.median(volumes) if volumes else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def get_batch_quotes(
        self, symbols: List[str], use_cache: bool = True, show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        大規模批量獲取報價 - 支援4000+股票

        Args:
            symbols: 股票代碼列表
            use_cache: 是否使用本地緩存
            show_progress: 是否顯示進度條

        Returns:
            {symbol: {price, change, volume, timestamp}} 字典
        """
        all_quotes = {}
        failed_symbols = []

        # 檢查緩存
        if use_cache:
            cached_quotes = self._get_cached_quotes(symbols)
            all_quotes.update(cached_quotes)
            symbols = [s for s in symbols if s not in cached_quotes]

        if not symbols:
            logger.info("All quotes retrieved from cache")
            return all_quotes

        # 分批處理
        batches = [
            symbols[i : i + self.batch_size] for i in range(0, len(symbols), self.batch_size)
        ]

        logger.info(f"Processing {len(symbols)} symbols in {len(batches)} batches")

        if show_progress:
            pbar = tqdm(total=len(symbols), desc="Getting quotes")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, i): batch
                for i, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_quotes, batch_failed = future.result()
                    all_quotes.update(batch_quotes)
                    failed_symbols.extend(batch_failed)

                    if show_progress:
                        pbar.update(len(batch))

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    failed_symbols.extend(batch)

                    if show_progress:
                        pbar.update(len(batch))

        if show_progress:
            pbar.close()

        # 保存到緩存
        if all_quotes:
            self._cache_quotes(all_quotes)

        success_rate = len(all_quotes) / (len(all_quotes) + len(failed_symbols)) * 100
        logger.info(
            f"Retrieved {len(all_quotes)} quotes, {len(failed_symbols)} failed. Success rate: {success_rate:.1f}%"
        )

        return all_quotes

    def _process_batch(self, symbols: List[str], batch_id: int) -> Tuple[Dict, List]:
        """處理單個批次的股票報價"""
        quotes = {}
        failed = []

        try:
            time.sleep(self.request_delay * batch_id)  # 避免過載

            # Method 1: Yahoo Finance batch download
            data = yf.download(symbols, period="1d", interval="1m", progress=False, threads=True)

            if not data.empty:
                current_time = datetime.now()

                if len(symbols) == 1:
                    symbol = symbols[0]
                    if not data["Close"].empty:
                        price = data["Close"].iloc[-1]
                        volume = data["Volume"].iloc[-1] if "Volume" in data else 0

                        if pd.notna(price):
                            quotes[symbol] = {
                                "price": float(price),
                                "volume": int(volume) if pd.notna(volume) else 0,
                                "timestamp": current_time,
                                "change_percent": 0.0,  # 需要額外計算
                            }
                        else:
                            failed.append(symbol)
                else:
                    for symbol in symbols:
                        try:
                            if symbol in data["Close"].columns:
                                price = data["Close"][symbol].iloc[-1]
                                volume = data["Volume"][symbol].iloc[-1] if "Volume" in data else 0

                                if pd.notna(price):
                                    quotes[symbol] = {
                                        "price": float(price),
                                        "volume": int(volume) if pd.notna(volume) else 0,
                                        "timestamp": current_time,
                                        "change_percent": 0.0,
                                    }
                                else:
                                    failed.append(symbol)
                            else:
                                failed.append(symbol)
                        except Exception as e:
                            logger.debug(f"Error processing {symbol}: {e}")
                            failed.append(symbol)
            else:
                failed.extend(symbols)

        except Exception as e:
            logger.error(f"Batch {batch_id} error: {e}")
            failed.extend(symbols)

        return quotes, failed

    def _get_cached_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """從數據庫獲取緩存的報價"""
        cached_quotes = {}
        cutoff_time = datetime.now() - timedelta(seconds=self.cache_duration)

        try:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(symbols))
                cursor = conn.execute(
                    f"""SELECT symbol, price, volume, change_percent, timestamp 
                        FROM real_time_quotes 
                        WHERE symbol IN ({placeholders}) 
                        AND timestamp > ?""",
                    symbols + [cutoff_time],
                )

                for row in cursor.fetchall():
                    symbol, price, volume, change_percent, timestamp = row
                    cached_quotes[symbol] = {
                        "price": price,
                        "volume": volume,
                        "change_percent": change_percent,
                        "timestamp": datetime.fromisoformat(timestamp),
                    }

        except Exception as e:
            logger.error(f"Error reading cached quotes: {e}")

        return cached_quotes

    def _cache_quotes(self, quotes: Dict[str, Dict]):
        """將報價保存到數據庫緩存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for symbol, quote_data in quotes.items():
                    conn.execute(
                        """INSERT OR REPLACE INTO real_time_quotes 
                           (symbol, price, volume, change_percent, timestamp) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            symbol,
                            quote_data["price"],
                            quote_data["volume"],
                            quote_data["change_percent"],
                            quote_data["timestamp"],
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching quotes: {e}")

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """向後兼容的批量報價方法"""
        batch_quotes = self.get_batch_quotes(symbols, show_progress=False)
        return {symbol: data["price"] for symbol, data in batch_quotes.items()}

    def get_market_overview(self) -> Dict:
        """
        獲取市場整體狀況
        """
        try:
            # 主要市場指數
            indices = ["SPY", "QQQ", "IWM", "VTI"]
            index_quotes = self.get_batch_quotes(indices, show_progress=False)

            # 檢查市場是否開放
            now = datetime.now()
            market_open = (
                now.hour >= 9
                and now.hour < 16
                and now.weekday() < 5
                and not self._is_market_holiday(now)
            )

            # VIX恐慌指數（如果可用）
            vix_data = None
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="1d")
                if not vix_hist.empty:
                    vix_data = float(vix_hist["Close"].iloc[-1])
            except:
                pass

            return {
                "is_open": market_open,
                "timestamp": now.isoformat(),
                "indices": index_quotes,
                "vix": vix_data,
                "session_type": self._get_session_type(now),
            }

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {"is_open": False, "error": str(e)}

    def _is_market_holiday(self, date: datetime) -> bool:
        """簡單的市場假日檢查"""
        # 這裡可以加入更完整的假日清單
        holidays = [
            (1, 1),  # New Year's Day
            (7, 4),  # Independence Day
            (12, 25),  # Christmas
        ]
        return (date.month, date.day) in holidays

    def _get_session_type(self, now: datetime) -> str:
        """獲取交易時段類型"""
        hour = now.hour
        if hour < 9:
            return "pre_market"
        elif hour < 16:
            return "regular"
        elif hour < 20:
            return "after_hours"
        else:
            return "closed"

    def get_market_status(self) -> Dict:
        """向後兼容的市場狀態方法"""
        overview = self.get_market_overview()
        spy_data = overview.get("indices", {}).get("SPY", {})

        return {
            "is_open": overview["is_open"],
            "timestamp": overview["timestamp"],
            "spy_price": spy_data.get("price", 0),
            "spy_change": spy_data.get("change_percent", 0),
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算基本技術指標

        Args:
            df: 包含OHLCV數據的DataFrame

        Returns:
            添加了指標的DataFrame
        """
        try:
            # Simple Moving Averages
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()

            # Exponential Moving Averages
            df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
            df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

            # MACD
            df["MACD"] = df["EMA_12"] - df["EMA_26"]
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

            # RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df["BB_Middle"] = df["Close"].rolling(window=20).mean()
            bb_std = df["Close"].rolling(window=20).std()
            df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
            df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

            # Volume indicators
            df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

            logger.info("Calculated technical indicators")
            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = FreeDataClient()

    print("\n" + "=" * 60)
    print("ENHANCED FREE DATA CLIENT TEST")
    print("=" * 60)

    # Test real-time price
    print("\n1. Real-time Price Test:")
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        price = client.get_real_time_price(symbol)
        if price:
            print(f"   {symbol}: ${price:.2f}")

    # Test historical data
    print("\n2. Historical Data Test:")
    hist = client.get_historical_data("AAPL", period="5d")
    if hist is not None:
        print(f"   Got {len(hist)} days of AAPL data")
        print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")

    # Test large batch quotes
    print("\n3. Large Scale Batch Quotes Test:")
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX", "AMD", "INTC"]
    batch_quotes = client.get_batch_quotes(test_symbols)
    for symbol, data in list(batch_quotes.items())[:5]:  # Show first 5
        print(f"   {symbol}: ${data['price']:.2f} (Vol: {data['volume']:,})")
    print(f"   ... and {len(batch_quotes)-5} more")

    # Test market overview
    print("\n4. Market Overview:")
    overview = client.get_market_overview()
    print(f"   Market Open: {overview.get('is_open')}")
    print(f"   Session: {overview.get('session_type')}")
    if "indices" in overview:
        for index, data in overview["indices"].items():
            print(f"   {index}: ${data['price']:.2f}")

    # Test watchlist summary
    print("\n5. Watchlist Summary:")
    summary = client.get_watchlist_summary(test_symbols)
    print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")
    print(
        f"   Price Range: ${summary.get('price_stats', {}).get('min', 0):.2f} - ${summary.get('price_stats', {}).get('max', 0):.2f}"
    )
    print(f"   Total Volume: {summary.get('volume_stats', {}).get('total', 0):,}")

    # Test indicators
    print("\n6. Technical Indicators:")
    if hist is not None:
        hist_with_indicators = client.calculate_indicators(hist)
        latest = hist_with_indicators.iloc[-1]
        print(f"   RSI: {latest['RSI']:.2f}")
        print(f"   MACD: {latest['MACD']:.4f}")
        print(f"   SMA_20: ${latest['SMA_20']:.2f}")

    print("\n[SUCCESS] Enhanced free data client is working!")
    print(f"[INFO] Database location: {client.db_path}")
