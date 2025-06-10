# core/config.py

import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# 為了讓輔助函式能使用 logger，我們將 logger 的初始化提前
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# --- 輔助函式：從檔案讀取股票列表 ---
def _load_symbols_from_file(filepath: str) -> List[str]:
    """從一個文字檔案讀取股票列表，每行一個。"""
    try:
        with open(filepath, 'r') as f:
            # 讀取每一行，去除空白，並過濾掉空行
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except FileNotFoundError:
        logger.warning(f"Ticker file not found at '{filepath}'. Using default example symbols.")
        # 如果檔案不存在，則返回一個小的預設列表以供測試
        return ["AAPL", "MSFT", "GOOG"]
# ------------------------------------

# 從 .env 檔案載入環境變數
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
TICKERS_FILE_PATH = "tickers.txt" # 假設檔案在專案根目錄
SYMBOLS_TO_TRADE: List[str] = _load_symbols_from_file(TICKERS_FILE_PATH)
# --- 從檔案動態載入股票列表 ---


# Risk Management Configuration (Example)
MAX_ORDER_QUANTITY_PER_TRADE: int = 100
MAX_POSITION_VALUE_PER_SYMBOL: float = 10000.00
INITIAL_CAPITAL_EXAMPLE: float = 100000.00

# Strategy Configuration (Example)
STRATEGIES_CONFIG: List[Dict[str, Any]] = [
    {
        "id": "SMA_Crossover_All_Stocks",
        "class_name": "MovingAverageCrossoverStrategy",
        "symbols": SYMBOLS_TO_TRADE, # 將讀取到的所有股票應用於此策略
        "params": {"short_window": 10, "long_window": 20}
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