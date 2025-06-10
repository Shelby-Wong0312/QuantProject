# core/config.py

import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

def _load_symbols_from_file(filepath: str) -> List[str]:
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except FileNotFoundError:
        logger.warning(f"Ticker file not found at '{filepath}'. Using default example symbols.")
        return ["AAPL", "MSFT", "GOOG"]

load_dotenv()

# Alpaca API Credentials and Configuration
ALPACA_API_KEY_ID: Optional[str] = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET_KEY: Optional[str] = os.getenv("ALPACA_SECRET_KEY")
POLYGON_API_KEY: Optional[str] = os.getenv("POLYGON_API_KEY")
ALPACA_PAPER_TRADING: bool = os.getenv("ALPACA_PAPER_TRADING", "True").lower() == "true"
ALPACA_API_BASE_URL: str = ("https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING else "https://api.alpaca.markets")
ALPACA_DATA_URL: str = ("wss://stream.data.sandbox.alpaca.markets/v2/iex" if ALPACA_PAPER_TRADING else "wss://stream.data.alpaca.markets/v2/iex")

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# --- 從檔案動態載入股票列表 ---
TICKERS_FILE_PATH = "tickers.txt"
SYMBOLS_TO_TRADE: List[str] = _load_symbols_from_file(TICKERS_FILE_PATH)
# --------------------------------

# Risk Management Configuration (Example)
MAX_ORDER_QUANTITY_PER_TRADE: int = 100
MAX_POSITION_VALUE_PER_SYMBOL: float = 10000.00
INITIAL_CAPITAL_EXAMPLE: float = 100000.00

# Strategy Configuration (Example)
STRATEGIES_CONFIG: List[Dict[str, Any]] = [
    {
        "id": "SMA_Crossover_All_Stocks",
        "class_name": "MovingAverageCrossoverStrategy",
        "symbols": SYMBOLS_TO_TRADE,
        # vvvvvv 修正此處的參數 vvvvvv
        "params": {"short_window": 5, "long_window": 20}
        # ^^^^^^ 修正此處的參數 ^^^^^^
    }
]

# Queue Configuration
MARKET_DATA_QUEUE_MAX_SIZE: int = 10000
SIGNAL_QUEUE_MAX_SIZE: int = 1000
ORDER_QUEUE_MAX_SIZE: int = 1000
FILL_QUEUE_MAX_SIZE: int = 1000
SYSTEM_CONTROL_QUEUE_MAX_SIZE: int = 100

# 確保 API 金鑰已設定
if ALPACA_API_KEY_ID is None or ALPACA_SECRET_KEY is None:
    raise ValueError("ALPACA_API_KEY_ID and ALPACA_SECRET_KEY must be set in .env file or environment variables.")
if POLYGON_API_KEY is None:
    raise ValueError("POLYGON_API_KEY must be set in .env file or environment variables.")