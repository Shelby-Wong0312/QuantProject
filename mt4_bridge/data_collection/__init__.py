# -*- coding: utf-8 -*-
"""
MT4 數據收集系統

本模組提供完整的 MT4 數據收集、聚合和存儲解決方案，包括：
- 實時 Tick 數據收集
- OHLC K線數據聚合
- 技術指標計算
- 高效數據存儲管理
- 與現有事件驅動架構的整合

主要組件：
- TickCollector: Tick 級數據收集器
- OHLCAggregator: OHLC 數據聚合器
- MT4DataFeed: 與事件系統整合的數據饋送器
- DataStorage: 高效數據存儲管理系統

使用示例：
    from mt4_bridge.data_collection import MT4DataFeed, TimeFrame
    from core.event_loop import EventLoop

    # 創建事件循環
    event_loop = EventLoop()

    # 創建 MT4 數據饋送器
    data_feed = MT4DataFeed(
        ["EURUSD", "GBPUSD"],
        event_queue=event_loop,
        timeframes=[TimeFrame.M1, TimeFrame.M5, TimeFrame.M15],
        enable_tick_collection=True,
        enable_indicators=True
    )

    # 啟動數據收集
    await data_feed.run()
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading System"

# 導入主要類和枚舉
from .tick_collector import TickCollector, TickData
from .ohlc_aggregator import OHLCAggregator, OHLCBar, TimeFrame, TechnicalIndicators
from .mt4_data_feed import MT4DataFeed
from .data_storage import DataStorage

# 導入常用的枚舉值，方便使用
from .ohlc_aggregator import TimeFrame

# 公開的 API
__all__ = [
    # 主要類
    "TickCollector",
    "OHLCAggregator",
    "MT4DataFeed",
    "DataStorage",
    # 數據結構
    "TickData",
    "OHLCBar",
    # 枚舉
    "TimeFrame",
    # 工具類
    "TechnicalIndicators",
]

# 模組級別的默認配置
DEFAULT_CONFIG = {
    "tick_collection": {
        "cache_size": 10000,
        "storage_path": "./data/mt4_ticks",
        "auto_save_interval": 300,  # 5分鐘
        "enable_csv": True,
        "enable_parquet": True,
    },
    "ohlc_aggregation": {
        "default_timeframes": [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1],
        "enable_indicators": True,
        "max_bars_per_timeframe": 1000,
        "indicator_periods": {
            "sma_short": 20,
            "sma_long": 50,
            "ema_short": 12,
            "ema_long": 26,
            "rsi": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
    },
    "data_storage": {
        "storage_path": "./data/mt4_storage",
        "enable_sqlite": True,
        "enable_parquet": True,
        "enable_csv": False,
        "enable_compression": True,
        "max_memory_cache": 100000,
        "auto_cleanup_days": 30,
        "batch_size": 1000,
        "flush_interval": 300,
    },
    "data_feed": {"market_event_timeframe": TimeFrame.M1, "price_history_length": 200},
}


def get_default_config():
    """獲取默認配置"""
    return DEFAULT_CONFIG.copy()


def create_mt4_data_collection_system(symbols, event_queue, config=None):
    """
    創建完整的 MT4 數據收集系統

    Args:
        symbols: 交易品種列表
        event_queue: 事件隊列
        config: 自定義配置 (可選)

    Returns:
        MT4DataFeed: 配置好的數據饋送器實例
    """
    if config is None:
        config = get_default_config()

    # 合併配置
    tick_config = config.get("tick_collection", {})
    ohlc_config = config.get("ohlc_aggregation", {})
    storage_config = config.get("data_storage", {})
    feed_config = config.get("data_feed", {})

    # 創建數據饋送器
    data_feed = MT4DataFeed(
        symbols,
        event_queue=event_queue,
        timeframes=ohlc_config.get(
            "default_timeframes", DEFAULT_CONFIG["ohlc_aggregation"]["default_timeframes"]
        ),
        enable_tick_collection=True,
        enable_indicators=ohlc_config.get("enable_indicators", True),
        market_event_timeframe=feed_config.get("market_event_timeframe", TimeFrame.M1),
        price_history_length=feed_config.get("price_history_length", 200),
    )

    return data_feed


# 版本信息
def get_version_info():
    """獲取版本信息"""
    return {
        "version": __version__,
        "author": __author__,
        "components": {
            "TickCollector": "實時 Tick 數據收集器",
            "OHLCAggregator": "OHLC 數據聚合器",
            "MT4DataFeed": "事件驅動數據饋送器",
            "DataStorage": "高效數據存儲系統",
        },
        "supported_timeframes": [tf.value for tf in TimeFrame],
        "supported_indicators": [
            "SMA (Simple Moving Average)",
            "EMA (Exponential Moving Average)",
            "RSI (Relative Strength Index)",
            "Bollinger Bands",
            "MACD (Moving Average Convergence Divergence)",
        ],
    }


# 兼容性檢查
def check_dependencies():
    """檢查依賴項是否可用"""
    dependencies = {
        "pandas": "數據處理",
        "numpy": "數值計算",
        "sqlite3": "SQLite 數據庫",
        "asyncio": "異步 I/O",
        "zmq": "ZeroMQ 通訊",
        "pathlib": "路徑處理",
    }

    missing = []
    available = {}

    for dep, description in dependencies.items():
        try:
            __import__(dep)
            available[dep] = description
        except ImportError:
            missing.append(dep)

    return {"available": available, "missing": missing, "all_available": len(missing) == 0}


# 模組初始化時檢查依賴
import logging

logger = logging.getLogger(__name__)

_deps = check_dependencies()
if not _deps["all_available"]:
    logger.warning(f"缺少依賴項: {_deps['missing']}")
    logger.warning("某些功能可能無法正常工作")
else:
    logger.debug("所有依賴項檢查通過")

# 清理臨時變量
del _deps, logging, logger
