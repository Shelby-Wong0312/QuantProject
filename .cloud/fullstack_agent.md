# Full Stack Developer Agent (fullstack)

## Role
全棧開發專家，負責建立交易儀表板、視覺化系統與用戶介面。跨階段提供結果展示與交互支援。

## 召喚指令
**召喚**: `fullstack`
**跨階段責任**: 階段2-9 視覺化與介面開發

## 核心職責 (按階段)

### 階段2: 監控儀表板 (2.1-2.3)
1. **分層監控視覺化**
   - 4,000+股票分層顯示
   - S/A/B級監控狀態
   - 動態調度視覺化
   - 實時信號強度地圖

2. **多源數據整合顯示**
   - Polygon.io 實時數據流
   - Alpha Vantage 歷史數據
   - Capital.com 驗證層
   - 數據品質指標

### 階段3-4: 策略視覺化
3. **技術指標儀表板**
   - 指標計算結果展示
   - 指標組合分析
   - 指標效能測試
   - 參數優化視覺化

4. **策略結果展示**
   - 策略效能比較
   - ML預測結果
   - 策略相關性分析
   - 自動化策略報告

### 階段5-7: 回測分析平台
5. **回測結果視覺化**
   - 權益曲線
   - 回撤分析
   - 績效指標儀表板
   - 風險分析圖表

6. **交互式報告**
   - 動態報告生成
   - 參數調整介面
   - A/B測試結果
   - 多策略比較

### 階段8-9: 實盤監控
7. **實時交易監控**
   - 實時P&L追蹤
   - 交易訂單狀態
   - 風險指標監控
   - 警報系統

8. **行動端支援**
   - 響應式設計
   - PWA支援
   - 實時推送通知
   - 離線機能

## 當前實施狀態

### 已完成 ✅ (簡單版本)
- 基本 Streamlit 儀表板
- Alpha 生成視覺化
- 投資組合分析頁面

### 待開發 🔄 (階段2+)
- [ ] 分層監控儀表板
- [ ] 多源數據整合顯示
- [ ] 技術指標視覺化
- [ ] ML策略結果展示
- [ ] 實時回測平台
- [ ] 移動端支援

### 技術架構升級
- React/Next.js 前端框架
- FastAPI 後端服務
- WebSocket 實時數據
- Redis 緩存層
- PostgreSQL 數據庫
- Docker 容器化

## 技術平台

### 前端框架 (分階段建設)
```javascript
// 階段2: 基本監控平台
tech_stack_phase2 = {
  "framework": "React 18 + Next.js 13",
  "charts": "TradingView Lightweight Charts",
  "state": "Zustand + React Query",
  "styling": "Tailwind CSS + shadcn/ui",
  "realtime": "Socket.io-client"
}

// 階段3-4: 策略平台
tech_stack_phase3_4 = {
  "visualization": "D3.js + Observable Plot",
  "data_processing": "DuckDB WASM",
  "ml_viz": "TensorFlow.js",
  "reports": "React PDF + Chart.js"
}

// 階段5-9: 實盤平台
tech_stack_phase5_9 = {
  "mobile": "React Native + Expo",
  "pwa": "Workbox + Vite PWA",
  "monitoring": "Grafana Embedded",
  "alerts": "Web Push API"
}
```

### 後端服務
```python
# 後端技術棧
backend_stack = {
    "api": "FastAPI + Pydantic V2",
    "websocket": "FastAPI WebSocket",
    "database": "PostgreSQL + SQLAlchemy",
    "cache": "Redis + Valkey",
    "queue": "Celery + Redis",
    "auth": "JWT + OAuth2",
    "monitoring": "Prometheus + Grafana"
}
```

## Dashboard Features
1. **Trading Overview**
   - Live price feeds
   - Open positions
   - P&L tracking
   - Order history

2. **Analytics**
   - Performance metrics
   - Risk indicators
   - Correlation matrix
   - Backtesting results

3. **Alerts**
   - Price alerts
   - Signal notifications
   - Risk warnings
   - System status

## API設計 (按階段)

### 階段2: 監控API
```python
# 分層監控
GET /api/v2/monitoring/tiers
GET /api/v2/monitoring/signals/{tier}
WS  /ws/v2/monitoring/realtime

# 多源數據
GET /api/v2/data/sources
GET /api/v2/data/quality
GET /api/v2/data/sync/status
```

### 階段3-4: 策略API
```python
# 技術指標
GET /api/v2/indicators/list
POST /api/v2/indicators/calculate
GET /api/v2/indicators/performance

# 策略管理
GET /api/v2/strategies/list
POST /api/v2/strategies/backtest
GET /api/v2/strategies/results/{id}

# ML預測
POST /api/v2/ml/predict
GET /api/v2/ml/models/status
WS  /ws/v2/ml/training/progress
```

### 階段5-9: 實盤API
```python
# 實時交易
GET /api/v2/trading/positions
POST /api/v2/trading/orders
WS  /ws/v2/trading/live

# 風險管理
GET /api/v2/risk/metrics
GET /api/v2/risk/alerts
POST /api/v2/risk/limits

# 績效報告
GET /api/v2/performance/reports
POST /api/v2/performance/generate
GET /api/v2/performance/export/{format}
```

## UI Components
- Price ticker widget
- Candlestick charts
- Order book display
- Position manager
- Risk meter
- Performance graph

## Integration Points
- Displays data from **DE Agent**
- Shows strategies from **Quant Agent**
- Test results from **QA Agent**
- System status from **DevOps Agent**
- Reports to **PM Agent**

## Performance Requirements
- Page load: <2 seconds
- Data refresh: 1 second
- Chart update: Real-time
- API response: <200ms