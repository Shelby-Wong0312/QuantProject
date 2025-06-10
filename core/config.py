# core/config.py

import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# 從 .env 檔案載入環境變數
load_dotenv()

# Alpaca API Credentials and Configuration
ALPACA_API_KEY_ID: Optional[str] = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET_KEY: Optional[str] = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER_TRADING: bool = os.getenv("ALPACA_PAPER_TRADING", "True").lower() == "true"

ALPACA_API_BASE_URL: str = (
    "https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING 
    else "https://api.alpaca.markets"
)

# vvvvvv 修正此處的 URL vvvvvv
ALPACA_DATA_URL: str = (
    "wss://stream.data.sandbox.alpaca.markets" if ALPACA_PAPER_TRADING
    else "wss://stream.data.alpaca.markets"
)
# ^^^^^^ 修正此處的 URL ^^^^^^

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# System Configuration
# 範例股票池，可從外部檔案載入或擴展。
# 若要涵蓋全市場，可使用 ["*"] (需 API 支援)
SYMBOLS_TO_TRADE: List[str] = ["AAPL", "MSFT", "GOOG"]

# Risk Management Configuration (Example)
MAX_ORDER_QUANTITY_PER_TRADE: int = 100
MAX_POSITION_VALUE_PER_SYMBOL: float = 10000.00
INITIAL_CAPITAL_EXAMPLE: float = 100000.00 # 用於 PortfolioManager 的初始資金範例

# Strategy Configuration (Example - can be a more complex structure or separate file)
STRATEGIES_CONFIG: List[Dict[str, Any]] = [
    {
        "id": "SMA_Cross_AAPL",
        "class_name": "MovingAverageCrossoverStrategy",
        "symbols": ["AAPL"],
        "params": {"short_window": 10, "long_window": 20}
    },
    {
        "id": "SMA_Cross_MSFT",
        "class_name": "MovingAverageCrossoverStrategy",
        "symbols": ["MSFT"],
        "params": {"short_window": 5, "long_window": 15}
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