# Cloud DE - Phase 4 任務指令書
## 生產環境部署任務
### 優先級: 🔴 緊急

---

## 📋 任務清單

### Task DE-401: Capital.com API 生產環境配置
**預計工時**: 1-2天  
**開始時間**: 立即

#### 具體步驟：
```python
# 1. 創建安全的憑證配置文件
# 檔案: config/api_credentials.json
{
    "capital_com": {
        "api_key": "YOUR_API_KEY",
        "api_secret": "YOUR_SECRET",
        "account_id": "YOUR_ACCOUNT_ID",
        "environment": "live",  // or "demo"
        "base_url": "https://api-capital.backend-capital.com"
    }
}

# 2. 實現憑證加密管理
# 檔案: src/api/auth_manager.py
import os
import json
from cryptography.fernet import Fernet
from typing import Dict

class AuthManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_credentials(self, credentials: Dict) -> bytes:
        """加密API憑證"""
        json_str = json.dumps(credentials)
        return self.cipher.encrypt(json_str.encode())
    
    def decrypt_credentials(self, encrypted: bytes) -> Dict:
        """解密API憑證"""
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
    
    def save_encrypted_credentials(self, filepath: str, credentials: Dict):
        """保存加密憑證"""
        encrypted = self.encrypt_credentials(credentials)
        with open(filepath, 'wb') as f:
            f.write(encrypted)

# 3. 更新 Capital.com 客戶端
# 檔案: src/api/capital_client.py
# 添加以下功能:
- OAuth 2.0 認證流程
- Token 自動刷新機制
- 請求重試邏輯
- 錯誤處理強化

# 4. 實現連接測試腳本
# 檔案: scripts/test_api_connection.py
async def test_connection():
    client = CapitalComClient()
    
    # 測試認證
    auth_status = await client.authenticate()
    print(f"Authentication: {'SUCCESS' if auth_status else 'FAILED'}")
    
    # 測試數據獲取
    market_data = await client.get_market_data('AAPL')
    print(f"Market Data: {market_data}")
    
    # 測試下單（使用最小數量）
    test_order = await client.place_test_order('AAPL', 'BUY', 1)
    print(f"Test Order: {test_order}")
    
    # 測試WebSocket
    await client.test_websocket_stream()
```

#### 驗收標準：
- ✅ API 成功認證
- ✅ 能獲取實時報價
- ✅ WebSocket 連接穩定
- ✅ 憑證安全加密存儲

---

### Task DE-402: 實時數據收集系統
**預計工時**: 2-3天  
**依賴**: DE-401 完成

#### 具體步驟：
```python
# 1. 實現多線程數據收集器
# 檔案: src/data/realtime_collector.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
import pandas as pd

class RealtimeDataCollector:
    def __init__(self, symbols: List[str], redis_host='localhost'):
        self.symbols = symbols
        self.redis_client = redis.Redis(host=redis_host)
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def collect_data(self):
        """並行收集多支股票數據"""
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.collect_symbol_data(symbol))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def collect_symbol_data(self, symbol: str):
        """收集單一股票數據"""
        # WebSocket 訂閱
        await self.subscribe_market_data(symbol)
        
        # 數據處理管道
        async for tick in self.data_stream(symbol):
            # 清洗數據
            cleaned = self.clean_tick_data(tick)
            
            # 存儲到 Redis
            self.redis_client.lpush(f"tick:{symbol}", json.dumps(cleaned))
            
            # 聚合到分鐘數據
            await self.aggregate_to_minute(symbol, cleaned)

# 2. 建立數據緩存層
# 檔案: src/data/cache_manager.py
class CacheManager:
    def __init__(self):
        self.redis = redis.Redis(decode_responses=True)
        self.mongo = MongoClient()['trading_db']
    
    def cache_tick_data(self, symbol: str, tick: Dict):
        """緩存tick數據"""
        key = f"tick:{symbol}:{datetime.now().timestamp()}"
        self.redis.setex(key, 3600, json.dumps(tick))  # 1小時過期
    
    def get_recent_ticks(self, symbol: str, minutes: int = 5):
        """獲取最近的tick數據"""
        pattern = f"tick:{symbol}:*"
        keys = self.redis.keys(pattern)
        # 返回最近N分鐘的數據

# 3. 數據驗證與清洗
# 檔案: src/data/data_validator.py
class DataValidator:
    def validate_tick(self, tick: Dict) -> bool:
        """驗證tick數據完整性"""
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        
        # 檢查必要字段
        if not all(field in tick for field in required_fields):
            return False
        
        # 驗證價格合理性
        if tick['price'] <= 0 or tick['price'] > 1000000:
            return False
        
        # 驗證時間戳
        if not self.is_valid_timestamp(tick['timestamp']):
            return False
        
        return True

# 4. 實現數據備份機制
# 檔案: scripts/backup_data.py
def backup_daily_data():
    """每日數據備份"""
    # 從 Redis 導出到文件
    # 壓縮並上傳到雲存儲
    # 清理過期數據
```

#### 驗收標準：
- ✅ 同時處理 50+ 股票數據流
- ✅ 數據延遲 <100ms
- ✅ 自動數據驗證與清洗
- ✅ 數據備份機制運作正常

---

### Task DE-403: 績效追蹤儀表板
**預計工時**: 3-4天  
**依賴**: DE-402 完成

#### 具體步驟：
```python
# 1. 使用 Streamlit 建立Web界面
# 檔案: dashboard/app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Quantitative Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 側邊欄控制
with st.sidebar:
    st.header("Control Panel")
    
    # 策略選擇
    strategy = st.selectbox(
        "Select Strategy",
        ["MPT Portfolio", "Day Trading", "Hybrid"]
    )
    
    # 時間範圍
    time_range = st.select_slider(
        "Time Range",
        options=["1D", "1W", "1M", "3M", "1Y", "ALL"]
    )
    
    # 自動刷新
    auto_refresh = st.checkbox("Auto Refresh (5s)")

# 主要內容區
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", "$125,430", "+12.5%")

with col2:
    st.metric("Daily P&L", "+$1,234", "+0.98%")

with col3:
    st.metric("Sharpe Ratio", "1.35", "+0.05")

with col4:
    st.metric("Win Rate", "65%", "+2%")

# 2. 實現實時圖表
# 檔案: dashboard/charts.py
def create_portfolio_chart(data: pd.DataFrame):
    """創建投資組合價值圖表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    # 添加基準線
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['benchmark'],
        mode='lines',
        name='S&P 500',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    return fig

# 3. 風險指標儀表板
# 檔案: dashboard/risk_dashboard.py
def risk_metrics_panel():
    """風險指標面板"""
    st.header("Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR 圖表
        var_chart = create_var_chart()
        st.plotly_chart(var_chart)
    
    with col2:
        # 持倉集中度
        concentration = calculate_concentration()
        st.progress(concentration)
        st.caption(f"Position Concentration: {concentration:.0%}")

# 4. 交易歷史查詢
# 檔案: dashboard/trade_history.py
def trade_history_table():
    """交易歷史表格"""
    trades = load_recent_trades()
    
    # 篩選器
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_filter = st.multiselect("Symbols", trades['symbol'].unique())
    
    with col2:
        side_filter = st.selectbox("Side", ["All", "BUY", "SELL"])
    
    with col3:
        date_filter = st.date_input("Date Range", [])
    
    # 顯示表格
    filtered_trades = apply_filters(trades, symbol_filter, side_filter, date_filter)
    st.dataframe(filtered_trades, use_container_width=True)
```

#### 驗收標準：
- ✅ 實時更新所有指標
- ✅ 圖表響應速度 <1秒
- ✅ 支持多策略切換
- ✅ 歷史數據查詢功能完整

---

## 📊 交付標準

### 整體要求：
1. **代碼質量**
   - 完整的錯誤處理
   - 詳細的日誌記錄
   - 單元測試覆蓋率 >80%

2. **性能要求**
   - API 響應時間 <200ms
   - 數據處理延遲 <100ms
   - 儀表板加載時間 <2秒

3. **安全要求**
   - API 憑證加密存儲
   - HTTPS 連接
   - 訪問控制機制

---

## 🚀 執行指令

```bash
# 1. 設置開發環境
pip install -r requirements_phase4.txt

# 2. 配置 Redis
docker run -d -p 6379:6379 redis:latest

# 3. 測試 API 連接
python scripts/test_api_connection.py

# 4. 啟動數據收集器
python src/data/realtime_collector.py --symbols TOP100

# 5. 運行儀表板
streamlit run dashboard/app.py
```

---

## 📅 時間線

- **Day 1**: API 配置與認證
- **Day 2**: WebSocket 測試與優化
- **Day 3-4**: 數據收集系統開發
- **Day 5**: 數據驗證與備份
- **Day 6-8**: 儀表板開發
- **Day 9**: 整合測試與優化

---

**任務分配人**: Cloud PM  
**執行人**: Cloud DE  
**開始時間**: 立即  
**截止時間**: 2025-08-19