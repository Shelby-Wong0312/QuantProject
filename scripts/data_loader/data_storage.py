"""
Data Storage and Indexing System
Cloud DE - Task DE-601
Optimized storage for 4,215 stocks with 15 years of data
"""

import pandas as pd
import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import redis
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for data storage"""

    use_postgres: bool = False  # Use PostgreSQL if available
    use_redis: bool = True  # Use Redis for caching
    use_parquet: bool = True  # Use Parquet for file storage
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading"
    postgres_user: str = "trader"
    postgres_password: str = "password"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # Cache TTL in seconds


class DataStorage:
    """Manages data storage across multiple backends"""

    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.sqlite_path = "data/historical_market_data.db"
        self.parquet_path = "data/parquet"
        self.postgres_conn = None
        self.redis_client = None

        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.parquet_path, exist_ok=True)

        # Initialize connections
        self._init_sqlite()
        if self.config.use_postgres:
            self._init_postgres()
        if self.config.use_redis:
            self._init_redis()

    def _init_sqlite(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.sqlite_path)

        # Create optimized tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                dividends REAL,
                stock_splits REAL,
                PRIMARY KEY (symbol, date)
            ) WITHOUT ROWID
        """
        )

        # Create partitioned tables for better performance
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data_2024 (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            ) WITHOUT ROWID
        """
        )

        # Create aggregated data table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data_daily_agg (
                date DATE PRIMARY KEY,
                total_volume INTEGER,
                avg_price REAL,
                market_cap REAL,
                active_symbols INTEGER
            )
        """
        )

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON market_data(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON market_data(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_close ON market_data(close)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_volume ON market_data(volume)")

        conn.commit()
        conn.close()
        logger.info("SQLite database initialized")

    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            self.postgres_conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )

            cursor = self.postgres_conn.cursor()

            # Create hypertable for time-series data (if using TimescaleDB)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data_ts (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    PRIMARY KEY (time, symbol)
                )
            """
            )

            # Try to create hypertable if TimescaleDB is available
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                cursor.execute(
                    "SELECT create_hypertable('market_data_ts', 'time', if_not_exists => TRUE)"
                )
                logger.info("TimescaleDB hypertable created")
            except Exception:
                logger.info("TimescaleDB not available, using standard PostgreSQL")

            self.postgres_conn.commit()
            logger.info("PostgreSQL database initialized")

        except Exception as e:
            logger.warning(f"PostgreSQL initialization failed: {e}")
            self.config.use_postgres = False

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False,  # Use binary for pickle
            )
            self.redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.config.use_redis = False

    def store_to_database(self, data: pd.DataFrame, table_name: str = "market_data"):
        """Store data to primary database"""
        if data.empty:
            return

        # Store to SQLite (always)
        self._store_to_sqlite(data, table_name)

        # Store to PostgreSQL if available
        if self.config.use_postgres and self.postgres_conn:
            self._store_to_postgres(data)

        # Store to Parquet if configured
        if self.config.use_parquet:
            self._store_to_parquet(data)

        # Update cache
        if self.config.use_redis and self.redis_client:
            self._update_cache(data)

    def _store_to_sqlite(self, data: pd.DataFrame, table_name: str):
        """Store data to SQLite"""
        conn = sqlite3.connect(self.sqlite_path)

        try:
            # Use transaction for better performance
            conn.execute("BEGIN TRANSACTION")

            # Prepare data
            data_to_store = data.copy()

            # Store in chunks for better memory management
            chunk_size = 10000
            for i in range(0, len(data_to_store), chunk_size):
                chunk = data_to_store.iloc[i : i + chunk_size]
                chunk.to_sql(table_name, conn, if_exists="append", index=False)

            conn.execute("COMMIT")
            logger.info(f"Stored {len(data)} rows to SQLite")

        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"SQLite storage failed: {e}")
            raise
        finally:
            conn.close()

    def _store_to_postgres(self, data: pd.DataFrame):
        """Store data to PostgreSQL"""
        if not self.postgres_conn:
            return

        cursor = self.postgres_conn.cursor()

        try:
            # Prepare data for bulk insert
            records = data.to_records(index=False)

            # Use COPY for fastest insertion
            data.columns.tolist()
            query = """
                INSERT INTO market_data_ts ({','.join(columns)})
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """

            # Execute in batches
            execute_batch(cursor, query, records, page_size=1000)
            self.postgres_conn.commit()

            logger.info(f"Stored {len(data)} rows to PostgreSQL")

        except Exception as e:
            self.postgres_conn.rollback()
            logger.error(f"PostgreSQL storage failed: {e}")

    def _store_to_parquet(self, data: pd.DataFrame):
        """Store data to Parquet files for efficient storage"""
        if data.empty:
            return

        # Group by symbol for partitioned storage
        for symbol in data["Symbol"].unique():
            symbol_data = data[data["Symbol"] == symbol]

            # Create year-based partitions
            symbol_data["year"] = pd.to_datetime(symbol_data["Date"]).dt.year

            for year in symbol_data["year"].unique():
                year_data = symbol_data[symbol_data["year"] == year]

                # Define file path
                file_path = os.path.join(
                    self.parquet_path, f"year={year}", f"{symbol}_{year}.parquet"
                )

                # Create directory if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Write to Parquet
                year_data.drop("year", axis=1).to_parquet(
                    file_path, engine="pyarrow", compression="snappy", index=False
                )

        logger.info("Stored data to Parquet files")

    def _update_cache(self, data: pd.DataFrame):
        """Update Redis cache with latest data"""
        if not self.redis_client:
            return

        try:
            # Cache latest price for each symbol
            for symbol in data["Symbol"].unique():
                symbol_data = data[data["Symbol"] == symbol].sort_values("Date")

                if not symbol_data.empty:
                    latest = symbol_data.iloc[-1]

                    # Create cache key
                    cache_key = f"latest:{symbol}"

                    # Store as JSON
                    cache_data = {
                        "symbol": symbol,
                        "date": (
                            latest["Date"].isoformat()
                            if hasattr(latest["Date"], "isoformat")
                            else str(latest["Date"])
                        ),
                        "open": float(latest["Open"]),
                        "high": float(latest["High"]),
                        "low": float(latest["Low"]),
                        "close": float(latest["Close"]),
                        "volume": int(latest["Volume"]),
                        "updated_at": datetime.now().isoformat(),
                    }

                    # Store with TTL
                    self.redis_client.setex(
                        cache_key, self.config.cache_ttl, json.dumps(cache_data)
                    )

            logger.info("Cache updated")

        except Exception as e:
            logger.error(f"Cache update failed: {e}")

    def get_data(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """Retrieve data for a symbol"""

        # Try cache first
        if self.config.use_redis and self.redis_client:
            cached = self._get_from_cache(symbol)
            if cached is not None:
                return cached

        # Try Parquet files (fastest for large queries)
        if self.config.use_parquet:
            self._get_from_parquet(symbol, start_date, end_date)
            if not data.empty:
                return data

        # Fall back to SQLite
        return self._get_from_sqlite(symbol, start_date, end_date)

    def _get_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from Redis cache"""
        if not self.redis_client:
            return None

        try:
            cache_key = f"data:{symbol}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                # Use JSON for safer deserialization
                import json

                try:
                    data_dict = json.loads(cached_data)
                    df = pd.DataFrame(data_dict)
                    logger.info(f"Retrieved {symbol} from cache")
                    return df
                except json.JSONDecodeError:
                    logger.error(f"Invalid cache data for {symbol}")
                    return None

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")

        return None

    def _get_from_parquet(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """Get data from Parquet files"""
        try:
            # Find relevant Parquet files
            all_data = []

            for year_dir in os.listdir(self.parquet_path):
                if year_dir.startswith("year="):
                    year = int(year_dir.split("=")[1])

                    # Check if year is in range
                    if start_date:
                        start_year = pd.to_datetime(start_date).year
                        if year < start_year:
                            continue

                    if end_date:
                        end_year = pd.to_datetime(end_date).year
                        if year > end_year:
                            continue

                    # Look for symbol file
                    file_path = os.path.join(
                        self.parquet_path, year_dir, f"{symbol}_{year}.parquet"
                    )

                    if os.path.exists(file_path):
                        df = pd.read_parquet(file_path)
                        all_data.append(df)

            if all_data:
                combined = pd.concat(all_data, ignore_index=True)

                # Filter by date range
                if start_date:
                    combined = combined[combined["Date"] >= start_date]
                if end_date:
                    combined = combined[combined["Date"] <= end_date]

                logger.info(f"Retrieved {symbol} from Parquet")
                return combined

        except Exception as e:
            logger.error(f"Parquet retrieval failed: {e}")

        return pd.DataFrame()

    def _get_from_sqlite(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """Get data from SQLite"""
        conn = sqlite3.connect(self.sqlite_path)

        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        logger.info(f"Retrieved {symbol} from SQLite")
        return df

    def create_data_catalog(self) -> pd.DataFrame:
        """Create a catalog of all available data"""
        conn = sqlite3.connect(self.sqlite_path)

        query = """
            SELECT 
                symbol,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as total_days,
                AVG(volume) as avg_volume,
                AVG(close) as avg_close,
                MIN(close) as min_close,
                MAX(close) as max_close
            FROM market_data
            GROUP BY symbol
            ORDER BY symbol
        """

        catalog = pd.read_sql_query(query, conn)
        conn.close()

        # Calculate additional metrics
        catalog["years_of_data"] = (
            pd.to_datetime(catalog["end_date"]) - pd.to_datetime(catalog["start_date"])
        ).dt.days / 365.25

        catalog["data_completeness"] = catalog["total_days"] / (
            catalog["years_of_data"] * 252
        )
        catalog["data_completeness"] = catalog["data_completeness"].clip(upper=1.0)

        # Format for display
        catalog["avg_volume"] = catalog["avg_volume"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
        )
        catalog["avg_close"] = catalog["avg_close"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        catalog["min_close"] = catalog["min_close"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        catalog["max_close"] = catalog["max_close"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        catalog["data_completeness"] = catalog["data_completeness"].apply(
            lambda x: f"{x*100:.1f}%"
        )

        # Save catalog
        catalog.to_csv("data/data_catalog.csv", index=False)
        logger.info(f"Data catalog created with {len(catalog)} symbols")

        return catalog

    def optimize_storage(self):
        """Optimize database storage"""
        conn = sqlite3.connect(self.sqlite_path)

        try:
            # Vacuum to reclaim space
            conn.execute("VACUUM")

            # Analyze for query optimization
            conn.execute("ANALYZE")

            # Rebuild indexes
            conn.execute("REINDEX")

            logger.info("Storage optimized")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
        finally:
            conn.close()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "sqlite_size": 0,
            "parquet_size": 0,
            "total_symbols": 0,
            "total_records": 0,
            "date_range": None,
        }

        # SQLite stats
        if os.path.exists(self.sqlite_path):
            stats["sqlite_size"] = os.path.getsize(self.sqlite_path) / (
                1024 * 1024
            )  # MB

            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
            stats["total_symbols"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM market_data")
            stats["total_records"] = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(date), MAX(date) FROM market_data")
            min_date, max_date = cursor.fetchone()
            stats["date_range"] = f"{min_date} to {max_date}"

            conn.close()

        # Parquet stats
        if os.path.exists(self.parquet_path):
            total_size = 0
            for root, dirs, files in os.walk(self.parquet_path):
                for file in files:
                    if file.endswith(".parquet"):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            stats["parquet_size"] = total_size / (1024 * 1024)  # MB

        return stats


if __name__ == "__main__":
    # Test the storage system
    storage = DataStorage()

    # Get storage statistics
    stats = storage.get_storage_stats()

    print("\n" + "=" * 60)
    print("DATA STORAGE SYSTEM")
    print("=" * 60)
    print(f"SQLite Database Size: {stats['sqlite_size']:.2f} MB")
    print(f"Parquet Files Size: {stats['parquet_size']:.2f} MB")
    print(f"Total Symbols: {stats['total_symbols']}")
    print(f"Total Records: {stats['total_records']:,}")
    print(f"Date Range: {stats['date_range']}")
    print("=" * 60)
