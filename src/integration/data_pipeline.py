"""
Data Pipeline - Manages data flow between components
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Manages data flow between different components of the trading system
    """

    def __init__(self, symbols: List[str], data_client: Any, buffer_size: int = 1000):
        """
        Initialize data pipeline

        Args:
            symbols: List of trading symbols
            data_client: Data client instance
            buffer_size: Size of data buffer for each symbol
        """
        self.symbols = symbols
        self.data_client = data_client
        self.buffer_size = buffer_size

        # Data buffers
        self.market_data_buffers = {symbol: deque(maxlen=buffer_size) for symbol in symbols}

        # Real-time subscriptions
        self.realtime_callbacks = {symbol: [] for symbol in symbols}

        # Data processing queues
        self.raw_data_queue = asyncio.Queue()
        self.processed_data_queue = asyncio.Queue()

        # Feature engineering parameters
        self.technical_indicators = [
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "BB_upper",
            "BB_lower",
            "ATR",
            "Volume_MA",
        ]

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.is_running = False

        # Data quality metrics
        self.data_quality_metrics = {
            "total_received": 0,
            "processed": 0,
            "errors": 0,
            "latency_ms": deque(maxlen=100),
        }

        logger.info(f"Data pipeline initialized for symbols: {symbols}")

    async def start(self):
        """Start the data pipeline"""
        self.is_running = True

        # Start data processing tasks
        asyncio.create_task(self._process_raw_data())
        asyncio.create_task(self._distribute_processed_data())

        logger.info("Data pipeline started")

    async def stop(self):
        """Stop the data pipeline"""
        self.is_running = False

        # Clear queues
        while not self.raw_data_queue.empty():
            await self.raw_data_queue.get()

        while not self.processed_data_queue.empty():
            await self.processed_data_queue.get()

        logger.info("Data pipeline stopped")

    async def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical data with feature engineering

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            # Fetch raw data
            raw_data = await self.data_client.get_historical_data(
                symbol=symbol, start_date=start_date, end_date=end_date, interval=interval
            )

            # Process and add features
            processed_data = await self._process_historical_data(raw_data)

            # Update buffer
            for _, row in processed_data.iterrows():
                self.market_data_buffers[symbol].append(row.to_dict())

            return processed_data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def subscribe_realtime(
        self, symbol: str, callback: Callable[[str, Dict[str, Any]], None]
    ):
        """
        Subscribe to real-time data updates

        Args:
            symbol: Trading symbol
            callback: Callback function for data updates
        """
        if symbol not in self.realtime_callbacks:
            self.realtime_callbacks[symbol] = []

        self.realtime_callbacks[symbol].append(callback)

        # Start streaming if not already active
        await self._start_realtime_stream(symbol)

        logger.info(f"Added real-time subscription for {symbol}")

    async def _start_realtime_stream(self, symbol: str):
        """Start real-time data stream for a symbol"""
        try:

            async def stream_handler(data: Dict[str, Any]):
                # Add to raw data queue
                await self.raw_data_queue.put(
                    {"symbol": symbol, "timestamp": datetime.now(), "data": data}
                )

                self.data_quality_metrics["total_received"] += 1

            # Subscribe to data client stream
            await self.data_client.subscribe_stream(symbol=symbol, callback=stream_handler)

        except Exception as e:
            logger.error(f"Error starting stream for {symbol}: {str(e)}")

    async def _process_raw_data(self):
        """Process raw data from queue"""
        while self.is_running:
            try:
                # Get raw data
                if not self.raw_data_queue.empty():
                    raw_item = await self.raw_data_queue.get()

                    start_time = datetime.now()

                    # Process in thread pool
                    processed_data = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._process_market_data,
                        raw_item["symbol"],
                        raw_item["data"],
                    )

                    # Calculate latency
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    self.data_quality_metrics["latency_ms"].append(latency)

                    # Add to processed queue
                    await self.processed_data_queue.put(
                        {
                            "symbol": raw_item["symbol"],
                            "timestamp": raw_item["timestamp"],
                            "data": processed_data,
                            "latency_ms": latency,
                        }
                    )

                    self.data_quality_metrics["processed"] += 1

                else:
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error processing raw data: {str(e)}")
                self.data_quality_metrics["errors"] += 1

    async def _distribute_processed_data(self):
        """Distribute processed data to subscribers"""
        while self.is_running:
            try:
                if not self.processed_data_queue.empty():
                    processed_item = await self.processed_data_queue.get()

                    symbol = processed_item["symbol"]

                    # Update buffer
                    self.market_data_buffers[symbol].append(processed_item["data"])

                    # Notify subscribers
                    for callback in self.realtime_callbacks.get(symbol, []):
                        try:
                            await callback(symbol, processed_item["data"])
                        except Exception as e:
                            logger.error(f"Error in callback for {symbol}: {str(e)}")
                else:
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error distributing data: {str(e)}")

    def _process_market_data(self, symbol: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw market data and add technical indicators

        Args:
            symbol: Trading symbol
            raw_data: Raw market data

        Returns:
            Processed data with features
        """
        try:
            # Convert to standardized format
            processed = {
                "symbol": symbol,
                "timestamp": raw_data.get("timestamp", datetime.now()),
                "open": float(raw_data.get("open", 0)),
                "high": float(raw_data.get("high", 0)),
                "low": float(raw_data.get("low", 0)),
                "close": float(raw_data.get("close", 0)),
                "volume": float(raw_data.get("volume", 0)),
            }

            # Get recent data from buffer
            buffer_data = list(self.market_data_buffers[symbol])

            if len(buffer_data) >= 50:  # Need enough data for indicators
                # Create DataFrame
                df = pd.DataFrame(buffer_data + [processed])

                # Calculate technical indicators
                processed.update(self._calculate_technical_indicators(df))

                # Add market microstructure features
                processed.update(self._calculate_microstructure_features(df))

            return processed

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            return raw_data

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        indicators = {}

        try:
            # Simple Moving Averages
            indicators["SMA_20"] = df["close"].rolling(window=20).mean().iloc[-1]
            indicators["SMA_50"] = df["close"].rolling(window=50).mean().iloc[-1]

            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators["RSI"] = (100 - (100 / (1 + rs))).iloc[-1]

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            indicators["MACD"] = (exp1 - exp2).iloc[-1]

            # Bollinger Bands
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            indicators["BB_upper"] = (sma + (std * 2)).iloc[-1]
            indicators["BB_lower"] = (sma - (std * 2)).iloc[-1]

            # ATR (Average True Range)
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators["ATR"] = true_range.rolling(window=14).mean().iloc[-1]

            # Volume MA
            indicators["Volume_MA"] = df["volume"].rolling(window=20).mean().iloc[-1]

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")

        return indicators

    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features"""
        features = {}

        try:
            # Price momentum
            features["momentum_1d"] = (
                (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) if len(df) > 1 else 0
            )
            features["momentum_5d"] = (
                (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) if len(df) > 5 else 0
            )

            # Volatility
            features["volatility_20d"] = df["close"].pct_change().rolling(window=20).std().iloc[-1]

            # Volume profile
            features["volume_ratio"] = df["volume"].iloc[-1] / features.get("Volume_MA", 1)

            # Price position
            features["price_position"] = (
                (
                    (df["close"].iloc[-1] - df["low"].rolling(window=20).min().iloc[-1])
                    / (
                        df["high"].rolling(window=20).max().iloc[-1]
                        - df["low"].rolling(window=20).min().iloc[-1]
                    )
                )
                if len(df) > 20
                else 0.5
            )

        except Exception as e:
            logger.error(f"Error calculating microstructure features: {str(e)}")

        return features

    async def _process_historical_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process historical data with feature engineering"""
        if raw_data.empty:
            return raw_data

        # Ensure we have OHLCV columns
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in raw_data.columns for col in required_columns):
            logger.error("Missing required OHLCV columns")
            return raw_data

        # Calculate all technical indicators
        processed_data = raw_data.copy()

        # Run calculations in thread pool
        processed_data = await asyncio.get_event_loop().run_in_executor(
            self.executor, self._add_all_features, processed_data
        )

        return processed_data

    def _add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to DataFrame"""
        # Technical indicators
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["BB_upper"] = sma + (std * 2)
        df["BB_lower"] = sma - (std * 2)

        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14).mean()

        # Volume indicators
        df["Volume_MA"] = df["volume"].rolling(window=20).mean()
        df["Volume_ratio"] = df["volume"] / df["Volume_MA"]

        # Price features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Forward fill any NaN values
        df = df.fillna(method="ffill").fillna(0)

        return df

    def prepare_features(self, data: Any) -> pd.DataFrame:
        """
        Prepare features for model input

        Args:
            data: Raw data (DataFrame or dict)

        Returns:
            DataFrame with prepared features
        """
        if isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Select relevant features
        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "BB_upper",
            "BB_lower",
            "ATR",
            "Volume_MA",
        ]

        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]

        return df[available_features]

    def get_buffer_data(self, symbol: str, n_samples: Optional[int] = None) -> List[Dict]:
        """
        Get buffered data for a symbol

        Args:
            symbol: Trading symbol
            n_samples: Number of samples to return (None for all)

        Returns:
            List of data dictionaries
        """
        buffer = self.market_data_buffers.get(symbol, deque())

        if n_samples is None:
            return list(buffer)
        else:
            return list(buffer)[-n_samples:]

    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        avg_latency = (
            np.mean(self.data_quality_metrics["latency_ms"])
            if self.data_quality_metrics["latency_ms"]
            else 0
        )

        return {
            "total_received": self.data_quality_metrics["total_received"],
            "processed": self.data_quality_metrics["processed"],
            "errors": self.data_quality_metrics["errors"],
            "success_rate": (
                self.data_quality_metrics["processed"]
                / max(1, self.data_quality_metrics["total_received"])
            ),
            "avg_latency_ms": avg_latency,
            "buffer_sizes": {
                symbol: len(buffer) for symbol, buffer in self.market_data_buffers.items()
            },
        }
