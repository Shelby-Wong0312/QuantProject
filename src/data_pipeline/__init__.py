# data/__init__.py
# 數據模組初始化文件

# 導入MT4數據收集器
from .mt4_data_collector import (
    MT4DataPipeline,
    MarketData,
    DataQualityChecker,
    DataCache as MT4DataCache,
    get_pipeline,
    create_pipeline,
    start_data_collection,
    stop_data_collection,
    get_realtime_data,
    get_historical_data
)

# 導入現有模組
try:
    from .capital_history_loader import CapitalHistoryLoader
except ImportError:
    CapitalHistoryLoader = None
    
try:
    from .data_manager import DataManager
except ImportError:
    DataManager = None
    
try:
    from .data_cache import DataCache
except ImportError:
    DataCache = None
    
try:
    from .history_loader import HistoryLoader
except ImportError:
    HistoryLoader = None
    
try:
    from .live_feed import LiveDataFeed
except ImportError:
    LiveDataFeed = None

__all__ = [
    # MT4數據收集
    'MT4DataPipeline',
    'MarketData',
    'DataQualityChecker',
    'MT4DataCache',
    'get_pipeline',
    'create_pipeline',
    'start_data_collection',
    'stop_data_collection',
    'get_realtime_data',
    'get_historical_data',
    
    # 現有模組
    'CapitalHistoryLoader',
    'DataManager',
    'DataCache',
    'HistoryLoader',
    'LiveDataFeed'
]