"""
PostgreSQL Database Setup for Capital.com Trading System
設置PostgreSQL數據庫用於存儲4,215個可交易股票的市場數據
"""

import os
import json
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
import logging
from dotenv import load_dotenv

# 設置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 加載環境變量
load_dotenv()


class DatabaseSetup:
    """PostgreSQL數據庫設置和管理"""

    def __init__(self):
        """初始化數據庫配置"""
        # 從環境變量讀取配置，如果沒有則使用默認值
        self.config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "quant_trading"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password"),
        }

        logger.info(
            f"Database config: {self.config['host']}:{self.config['port']}/{self.config['database']}"
        )

    def check_postgres_installed(self) -> bool:
        """檢查PostgreSQL是否已安裝"""
        try:
            # 嘗試連接到默認的postgres數據庫
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                database="postgres",
                user=self.config["user"],
                password=self.config["password"],
            )
            conn.close()
            logger.info("[OK] PostgreSQL is installed and running")
            return True
        except Exception as e:
            logger.error(f"[X] PostgreSQL not available: {e}")
            logger.info(
                "Please install PostgreSQL from: https://www.postgresql.org/download/"
            )
            return False

    def create_database(self) -> bool:
        """創建數據庫（如果不存在）"""
        try:
            # 連接到默認的postgres數據庫
            conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                database="postgres",
                user=self.config["user"],
                password=self.config["password"],
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # 檢查數據庫是否存在
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config["database"],),
            )

            if cursor.fetchone():
                logger.info(f"[OK] Database '{self.config['database']}' already exists")
            else:
                # 創建數據庫
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(self.config["database"])
                    )
                )
                logger.info(f"[OK] Created database '{self.config['database']}'")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"[X] Failed to create database: {e}")
            return False

    def create_tables(self) -> bool:
        """創建所有必要的表"""
        try:
            # 連接到目標數據庫
            conn = psycopg2.connect(**self.config)
            cursor = conn.cursor()

            # 1. 創建股票信息表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) UNIQUE NOT NULL,
                    epic VARCHAR(50),
                    name VARCHAR(255),
                    exchange VARCHAR(50),
                    currency VARCHAR(10),
                    is_tradable BOOLEAN DEFAULT TRUE,
                    last_validated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            logger.info("[OK] Created 'stocks' table")

            # 2. 創建Tick數據表（分區表）
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tick_data (
                    id BIGSERIAL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    bid DECIMAL(20, 8) NOT NULL,
                    ask DECIMAL(20, 8) NOT NULL,
                    last_price DECIMAL(20, 8),
                    spread DECIMAL(20, 8),
                    volume BIGINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, timestamp)
                ) PARTITION BY RANGE (timestamp)
            """
            )
            logger.info("[OK] Created 'tick_data' partitioned table")

            # 3. 創建OHLC數據表（分區表）
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id BIGSERIAL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    open_price DECIMAL(20, 8) NOT NULL,
                    high_price DECIMAL(20, 8) NOT NULL,
                    low_price DECIMAL(20, 8) NOT NULL,
                    close_price DECIMAL(20, 8) NOT NULL,
                    volume BIGINT DEFAULT 0,
                    tick_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                ) PARTITION BY RANGE (timestamp)
            """
            )
            logger.info("[OK] Created 'ohlc_data' partitioned table")

            # 4. 創建交易信號表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    strategy VARCHAR(100) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    price DECIMAL(20, 8),
                    quantity INTEGER,
                    confidence DECIMAL(5, 4),
                    metadata JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    executed BOOLEAN DEFAULT FALSE,
                    execution_time TIMESTAMP WITH TIME ZONE,
                    execution_price DECIMAL(20, 8),
                    pnl DECIMAL(20, 8)
                )
            """
            )
            logger.info("[OK] Created 'trading_signals' table")

            # 5. 創建回測結果表
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY,
                    strategy VARCHAR(100) NOT NULL,
                    symbols TEXT[],
                    start_date DATE,
                    end_date DATE,
                    initial_capital DECIMAL(20, 2),
                    final_capital DECIMAL(20, 2),
                    total_return DECIMAL(10, 4),
                    sharpe_ratio DECIMAL(10, 4),
                    max_drawdown DECIMAL(10, 4),
                    win_rate DECIMAL(5, 4),
                    total_trades INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            logger.info("[OK] Created 'backtest_results' table")

            # 6. 創建索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_stocks_tradable ON stocks(is_tradable)",
                "CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_tf_time ON ohlc_data(symbol, timeframe, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON trading_signals(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_signals_executed ON trading_signals(executed)",
                "CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy)",
            ]

            for idx_sql in indexes:
                cursor.execute(idx_sql)
            logger.info("[OK] Created all indexes")

            # 創建分區（按月）
            self._create_monthly_partitions(cursor)

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("[OK] All tables created successfully")
            return True

        except Exception as e:
            logger.error(f"[X] Failed to create tables: {e}")
            return False

    def _create_monthly_partitions(self, cursor):
        """創建月度分區表"""
        try:
            # 創建最近3個月和未來1個月的分區
            from datetime import datetime
            from dateutil.relativedelta import relativedelta

            current_date = datetime.now()

            for i in range(-3, 2):  # -3到+1月
                partition_date = current_date + relativedelta(months=i)
                year = partition_date.year
                month = partition_date.month

                # 分區名稱

                # 分區日期範圍
                start_date = partition_date.replace(day=1)
                if month == 12:
                    start_date.replace(year=year + 1, month=1)
                else:
                    start_date.replace(month=month + 1)

                # 創建tick數據分區
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS {tick_partition} 
                    PARTITION OF tick_data
                    FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') 
                    TO ('{end_date.strftime('%Y-%m-%d')}')
                """
                )

                # 創建OHLC數據分區
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS {ohlc_partition} 
                    PARTITION OF ohlc_data
                    FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') 
                    TO ('{end_date.strftime('%Y-%m-%d')}')
                """
                )

                logger.info(f"[OK] Created partitions for {year}-{month:02d}")

        except Exception as e:
            logger.warning(f"Partition creation note: {e}")

    def import_validated_stocks(self) -> bool:
        """導入已驗證的股票列表"""
        try:
            # 讀取已驗證的股票
            with open("TRADABLE_TICKERS.txt", "r") as f:
                tickers = [line.strip() for line in f if line.strip()]

            conn = psycopg2.connect(**self.config)
            cursor = conn.cursor()

            # 批量插入股票信息
            for ticker in tickers:
                cursor.execute(
                    """
                    INSERT INTO stocks (symbol, is_tradable)
                    VALUES (%s, %s)
                    ON CONFLICT (symbol) 
                    DO UPDATE SET 
                        is_tradable = EXCLUDED.is_tradable,
                        last_validated = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (ticker, True),
                )

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"[OK] Imported {len(tickers)} tradable stocks")
            return True

        except FileNotFoundError:
            logger.warning("TRADABLE_TICKERS.txt not found, skipping import")
            return True
        except Exception as e:
            logger.error(f"[X] Failed to import stocks: {e}")
            return False

    def test_connection(self) -> bool:
        """測試數據庫連接"""
        try:
            conn = psycopg2.connect(**self.config)
            cursor = conn.cursor()

            # 測試查詢
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_tradable = TRUE")
            count = cursor.fetchone()[0]

            logger.info("[OK] Database connection successful")
            logger.info(f"[STATS] Found {count} tradable stocks in database")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"[X] Connection test failed: {e}")
            return False

    def setup_all(self) -> bool:
        """執行完整的數據庫設置"""
        logger.info("=" * 60)
        logger.info("POSTGRESQL DATABASE SETUP")
        logger.info("=" * 60)

        # 1. 檢查PostgreSQL
        if not self.check_postgres_installed():
            return False

        # 2. 創建數據庫
        if not self.create_database():
            return False

        # 3. 創建表
        if not self.create_tables():
            return False

        # 4. 導入股票列表
        if not self.import_validated_stocks():
            return False

        # 5. 測試連接
        if not self.test_connection():
            return False

        logger.info("=" * 60)
        logger.info("[SUCCESS] Database setup completed successfully!")
        logger.info("=" * 60)

        # 保存配置
        self.save_config()

        return True

    def save_config(self):
        """保存數據庫配置到文件"""
        config_file = "db_config.json"
        config_data = {
            "database": self.config,
            "tables": {
                "stocks": "Stock information and validation status",
                "tick_data": "Real-time tick data (partitioned by month)",
                "ohlc_data": "OHLC candle data (partitioned by month)",
                "trading_signals": "Trading signals from strategies",
                "backtest_results": "Backtest performance metrics",
            },
            "created_at": datetime.now().isoformat(),
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"[OK] Configuration saved to {config_file}")


def main():
    """主函數"""
    setup = DatabaseSetup()

    # 檢查是否有.env文件
    if not os.path.exists(".env"):
        logger.warning("No .env file found. Creating default configuration...")
        with open(".env", "w") as f:
            f.write(
                """# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quant_trading
DB_USER=postgres
DB_PASSWORD=password

# Capital.com API (existing)
CAPITAL_API_KEY=your_api_key_here
CAPITAL_API_IDENTIFIER=your_identifier_here
CAPITAL_API_PASSWORD=your_password_here
CAPITAL_API_DEMO=true
"""
            )
        logger.info("[OK] Created .env file with default settings")
        logger.info("Please update the database password in .env file if needed")

    # 執行設置
    if setup.setup_all():
        print("\nNext steps:")
        print("1. Database is ready for data collection")
        print("2. Run 'python download_historical_data.py' to start downloading data")
        print("3. Use 'python src/capital_service.py' to connect to Capital.com")
    else:
        print("\nSetup failed. Please check the logs and try again.")
        print("Make sure PostgreSQL is installed and running.")


if __name__ == "__main__":
    main()
