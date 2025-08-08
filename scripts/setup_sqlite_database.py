"""
SQLite Database Setup for Capital.com Trading System
使用SQLite存儲4,215個可交易股票的市場數據（支持15年+歷史數據）
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import pandas as pd

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLiteSetup:
    """SQLite數據庫設置和管理"""
    
    def __init__(self):
        """初始化數據庫配置"""
        self.db_path = 'quant_trading.db'
        self.backup_path = 'quant_trading_backup.db'
        
        logger.info(f"Database path: {self.db_path}")
    
    def create_database(self) -> bool:
        """創建數據庫和所有表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 啟用外鍵約束
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # 設置性能優化
            cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store = MEMORY")
            
            # 1. 創建股票信息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    epic TEXT,
                    name TEXT,
                    exchange TEXT,
                    currency TEXT,
                    is_tradable INTEGER DEFAULT 1,
                    last_validated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("[OK] Created 'stocks' table")
            
            # 2. 創建OHLC數據表（主表，存儲所有時間框架）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    tick_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            logger.info("[OK] Created 'ohlc_data' table")
            
            # 3. 創建日線數據專用表（15年數據，優化查詢）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    adjusted_close REAL,
                    dividend REAL DEFAULT 0,
                    split_ratio REAL DEFAULT 1,
                    UNIQUE(symbol, date)
                )
            """)
            logger.info("[OK] Created 'daily_data' table")
            
            # 4. 創建分鐘數據表（短期高頻數據）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS minute_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    bid REAL,
                    ask REAL,
                    spread REAL,
                    UNIQUE(symbol, timestamp)
                )
            """)
            logger.info("[OK] Created 'minute_data' table")
            
            # 5. 創建Tick數據表（實時數據）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    last_price REAL,
                    spread REAL,
                    volume INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("[OK] Created 'tick_data' table")
            
            # 6. 創建交易信號表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    price REAL,
                    quantity INTEGER,
                    confidence REAL,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed INTEGER DEFAULT 0,
                    execution_time TIMESTAMP,
                    execution_price REAL,
                    pnl REAL
                )
            """)
            logger.info("[OK] Created 'trading_signals' table")
            
            # 7. 創建回測結果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    symbols TEXT,
                    start_date DATE,
                    end_date DATE,
                    initial_capital REAL,
                    final_capital REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("[OK] Created 'backtest_results' table")
            
            # 8. 創建數據統計表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    record_count INTEGER,
                    oldest_date DATE,
                    newest_date DATE,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe)
                )
            """)
            logger.info("[OK] Created 'data_stats' table")
            
            # 創建索引（優化查詢性能）
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_stocks_tradable ON stocks(is_tradable)",
                
                "CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_tf_time ON ohlc_data(symbol, timeframe, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ohlc_symbol ON ohlc_data(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_ohlc_timestamp ON ohlc_data(timestamp)",
                
                "CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON daily_data(symbol, date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_data(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data(date)",
                
                "CREATE INDEX IF NOT EXISTS idx_minute_symbol_time ON minute_data(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_minute_timestamp ON minute_data(timestamp)",
                
                "CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data(symbol, timestamp DESC)",
                
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON trading_signals(symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_signals_executed ON trading_signals(executed)",
                
                "CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy)",
                
                "CREATE INDEX IF NOT EXISTS idx_stats_symbol ON data_stats(symbol)"
            ]
            
            for idx_sql in indexes:
                cursor.execute(idx_sql)
            logger.info("[OK] Created all indexes")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("[OK] Database created successfully")
            return True
            
        except Exception as e:
            logger.error(f"[X] Failed to create database: {e}")
            return False
    
    def import_validated_stocks(self) -> bool:
        """導入已驗證的股票列表"""
        try:
            # 讀取已驗證的股票
            if not os.path.exists('TRADABLE_TICKERS.txt'):
                logger.warning("TRADABLE_TICKERS.txt not found")
                return True
            
            with open('TRADABLE_TICKERS.txt', 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 批量插入股票信息
            for ticker in tickers:
                cursor.execute("""
                    INSERT OR REPLACE INTO stocks (symbol, is_tradable)
                    VALUES (?, ?)
                """, (ticker, 1))
            
            conn.commit()
            
            # 統計
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_tradable = 1")
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            logger.info(f"[OK] Imported {count} tradable stocks")
            return True
            
        except Exception as e:
            logger.error(f"[X] Failed to import stocks: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """優化數據庫性能"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # VACUUM優化
            cursor.execute("VACUUM")
            
            # 分析表統計
            cursor.execute("ANALYZE")
            
            # 獲取數據庫大小
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            size_bytes = cursor.fetchone()[0]
            size_mb = size_bytes / (1024 * 1024)
            
            logger.info(f"[OK] Database optimized. Size: {size_mb:.2f} MB")
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"[X] Failed to optimize database: {e}")
            return False
    
    def test_connection(self) -> bool:
        """測試數據庫連接"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 測試查詢
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_tradable = 1")
            tradable_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ohlc_data")
            ohlc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM daily_data")
            daily_count = cursor.fetchone()[0]
            
            logger.info(f"[OK] Database connection successful")
            logger.info(f"[STATS] Tradable stocks: {tradable_count}")
            logger.info(f"[STATS] OHLC records: {ohlc_count}")
            logger.info(f"[STATS] Daily records: {daily_count}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"[X] Connection test failed: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """獲取數據庫信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            info = {}
            
            # 獲取所有表的記錄數
            tables = ['stocks', 'ohlc_data', 'daily_data', 'minute_data', 'tick_data', 
                     'trading_signals', 'backtest_results', 'data_stats']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                info[f'{table}_count'] = cursor.fetchone()[0]
            
            # 獲取數據庫大小
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            info['size_bytes'] = cursor.fetchone()[0]
            info['size_mb'] = info['size_bytes'] / (1024 * 1024)
            
            cursor.close()
            conn.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    def setup_all(self) -> bool:
        """執行完整的數據庫設置"""
        logger.info("=" * 60)
        logger.info("SQLITE DATABASE SETUP")
        logger.info("=" * 60)
        
        # 1. 創建數據庫
        if not self.create_database():
            return False
        
        # 2. 導入股票列表
        if not self.import_validated_stocks():
            return False
        
        # 3. 優化數據庫
        if not self.optimize_database():
            return False
        
        # 4. 測試連接
        if not self.test_connection():
            return False
        
        # 5. 顯示數據庫信息
        info = self.get_database_info()
        
        logger.info("=" * 60)
        logger.info("[SUCCESS] Database setup completed!")
        logger.info("=" * 60)
        logger.info("Database Information:")
        for key, value in info.items():
            if 'count' in key:
                logger.info(f"  {key}: {value:,}")
            elif 'mb' in key:
                logger.info(f"  {key}: {value:.2f} MB")
        
        # 保存配置
        self.save_config(info)
        
        return True
    
    def save_config(self, info: Dict):
        """保存數據庫配置到文件"""
        config_file = 'db_config.json'
        config_data = {
            'database': {
                'type': 'SQLite',
                'path': self.db_path,
                'backup_path': self.backup_path
            },
            'tables': {
                'stocks': 'Stock information and validation status',
                'daily_data': '15+ years of daily OHLC data',
                'minute_data': '4 weeks of minute-level data',
                'tick_data': 'Real-time tick data',
                'ohlc_data': 'Multi-timeframe OHLC data',
                'trading_signals': 'Trading signals from strategies',
                'backtest_results': 'Backtest performance metrics',
                'data_stats': 'Data coverage statistics'
            },
            'performance': {
                'journal_mode': 'WAL',
                'cache_size': '64MB',
                'indexes': 18
            },
            'statistics': info,
            'created_at': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"[OK] Configuration saved to {config_file}")


def main():
    """主函數"""
    setup = SQLiteSetup()
    
    # 執行設置
    if setup.setup_all():
        print("\n" + "=" * 60)
        print("Next steps:")
        print("1. Database is ready for data collection")
        print("2. Run 'python download_long_term_data.py' to download 15 years of data")
        print("3. The database supports:")
        print("   - 15+ years of daily data")
        print("   - 16 weeks of hourly data")
        print("   - 4 weeks of minute data")
        print("   - Real-time tick data")
        print("=" * 60)
    else:
        print("\nSetup failed. Please check the logs and try again.")


if __name__ == "__main__":
    main()