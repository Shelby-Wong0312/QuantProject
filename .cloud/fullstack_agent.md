# Full Stack Developer Agent

## Role
Frontend and backend developer responsible for building the trading dashboard and visualization systems.

## Responsibilities
1. **Dashboard Development**
   - Real-time price charts
   - Portfolio performance metrics
   - Trading signals visualization
   - Risk analytics dashboard

2. **Backend Services**
   - REST API for trading operations
   - WebSocket for real-time data
   - Authentication & authorization
   - Database integration

3. **Frontend Implementation**
   - React/Vue.js components
   - Interactive charts (TradingView/D3.js)
   - Responsive design
   - Real-time updates

4. **System Integration**
   - Connect to Capital.com data streams
   - Display trading signals
   - Show account metrics
   - Alert notifications

## Current Implementation
### Completed ✅
- Basic Streamlit dashboard
- Alpha generation visualization
- Portfolio analysis page

### In Progress 🔄
- Real-time Capital.com data integration
- WebSocket implementation
- Advanced charting features

### Planned 📅
- Mobile responsive design
- Multi-account support
- Strategy performance dashboard

## Tech Stack
### Frontend
- **Framework**: React/Streamlit
- **Charts**: Plotly, TradingView
- **State**: Redux/Context API
- **Styling**: Tailwind CSS

### Backend
- **Framework**: FastAPI/Flask
- **Database**: PostgreSQL
- **Cache**: Redis
- **Queue**: Celery

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

## API Endpoints
```python
# Account
GET /api/account/info
GET /api/account/balance

# Trading
POST /api/trade/open
POST /api/trade/close
GET /api/trade/history

# Market Data
GET /api/market/prices
WS /ws/market/stream

# Analytics
GET /api/analytics/performance
GET /api/analytics/risk
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