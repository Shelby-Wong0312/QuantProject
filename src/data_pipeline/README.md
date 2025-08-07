# 數據管道模組

## 概述
數據管道模組負責所有數據收集、處理和特徵工程功能，包括MT4實時數據收集、Capital.com API整合、數據質量檢查和技術指標計算。

## 核心組件

### 1. MT4數據收集系統 (`mt4_data_collector.py`)
完整的MT4實時數據收集和處理系統。

#### 主要功能：
- **實時Tick數據收集**：毫秒級tick數據流
- **多時間框架K線聚合**：M1, M5, M15, H1, H4, D1, W1, MN1
- **數據質量檢查**：點差、價格跳動、成交量驗證
- **內存緩存管理**：高效的數據緩存和查詢
- **持久化存儲**：SQLite數據庫存儲
- **技術指標計算**：SMA, EMA, RSI, Bollinger Bands等
- **統一數據接口**：標準化的MarketData結構

#### 使用範例：

```python
from src.data_pipeline import (
    MT4DataPipeline,
    start_data_collection,
    get_realtime_data,
    get_historical_data
)

# 方法1：快速啟動
pipeline = start_data_collection(['EURUSD', 'GBPUSD'])

# 方法2：自定義配置
pipeline = MT4DataPipeline(
    enable_storage=True,      # 啟用數據存儲
    enable_cache=True,        # 啟用內存緩存
    enable_quality_check=True # 啟用質量檢查
)

# 連接並啟動
if pipeline.connect():
    pipeline.start()
    pipeline.subscribe(['EURUSD', 'GBPUSD'])
    
    # 添加數據回調
    def on_market_data(data):
        print(f"{data.symbol}: Bid={data.bid}, Ask={data.ask}")
        if data.indicators:
            print(f"RSI={data.indicators.get('rsi14')}")
    
    pipeline.add_callback(on_market_data)
    
    # 獲取實時數據
    latest = pipeline.get_latest_data('EURUSD')
    
    # 獲取歷史K線
    df = pipeline.get_dataframe('EURUSD', TimeFrame.M5, periods=100)
    
    # 獲取技術指標
    indicators = pipeline.get_indicators('EURUSD', TimeFrame.M5)
    print(f"SMA20: {indicators.get('sma20')}")
    print(f"RSI14: {indicators.get('rsi14')}")
    
    # 獲取統計信息
    stats = pipeline.get_stats()
    print(f"總Tick數: {stats['total_ticks']}")
    print(f"有效率: {stats['validity_rate']:.2%}")
```

### 2. 數據質量檢查器 (`DataQualityChecker`)

自動檢查和過濾異常數據：

```python
from src.data_pipeline import DataQualityChecker

checker = DataQualityChecker(
    max_spread_ratio=0.01,  # 最大點差1%
    max_price_jump=0.05,    # 最大價格跳動5%
    min_volume=0            # 最小成交量
)

# 檢查tick數據
is_valid, error_msg = checker.check_tick(tick_data)
if not is_valid:
    print(f"數據異常: {error_msg}")
```

### 3. 數據緩存管理 (`DataCache`)

高效的內存數據緩存：

```python
from src.data_pipeline import MT4DataCache

cache = MT4DataCache(
    max_tick_cache=10000,  # 最大tick緩存
    max_ohlc_cache=5000    # 最大K線緩存
)

# 獲取最近數據
recent_ticks = cache.get_recent_ticks('EURUSD', count=100)
recent_bars = cache.get_recent_ohlc('EURUSD', TimeFrame.M5, count=50)

# 轉換為DataFrame
df = cache.to_dataframe('EURUSD', TimeFrame.M5)
```

### 4. 統一市場數據結構 (`MarketData`)

標準化的數據接口：

```python
@dataclass
class MarketData:
    symbol: str           # 交易品種
    timestamp: datetime   # 時間戳
    bid: float           # 買價
    ask: float           # 賣價
    mid: float           # 中間價
    spread: float        # 點差
    volume: int          # 成交量
    ohlc_1m: Dict        # 1分鐘K線
    ohlc_5m: Dict        # 5分鐘K線
    indicators: Dict     # 技術指標
```

## 技術指標

系統支援以下技術指標的實時計算：

- **移動平均**：SMA, EMA
- **震盪指標**：RSI
- **波動指標**：Bollinger Bands
- **自定義指標**：可擴展添加

## 數據存儲

### SQLite數據庫結構

```sql
-- Tick數據表
CREATE TABLE tick_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    timestamp DATETIME,
    bid REAL,
    ask REAL,
    spread REAL,
    volume INTEGER
);

-- OHLC數據表
CREATE TABLE ohlc_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    timestamp DATETIME,
    timeframe TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    tick_count INTEGER
);
```

## 性能優化

- **異步處理**：使用線程池處理數據
- **批量寫入**：減少數據庫IO
- **內存緩存**：熱數據保存在內存
- **數據壓縮**：歷史數據自動壓縮

## 錯誤處理

系統實現了完善的錯誤處理：

- 自動重連機制
- 數據質量過濾
- 異常數據記錄
- 統計信息追蹤

## 配置選項

```python
# 創建自定義配置的管道
pipeline = MT4DataPipeline(
    connector=custom_connector,      # 自定義連接器
    enable_storage=True,             # 啟用存儲
    enable_cache=True,               # 啟用緩存
    enable_quality_check=True        # 啟用質量檢查
)
```

## 測試

運行測試：

```bash
# 單元測試
python tests/test_mt4_data_pipeline.py

# 集成測試（需要MT4連接）
python tests/test_mt4_data_pipeline.py --integration
```

## 注意事項

1. **MT4連接**：確保MT4已啟動並載入PythonBridge EA
2. **數據質量**：自動過濾異常數據，但建議定期檢查
3. **存儲空間**：長期運行需要足夠的磁盤空間
4. **性能監控**：通過get_stats()監控系統性能

## 下一步

- 整合更多技術指標
- 實現數據壓縮和歸檔
- 添加更多數據源支援
- 優化高頻數據處理性能