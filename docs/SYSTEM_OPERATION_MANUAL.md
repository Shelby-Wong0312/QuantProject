# System Operation Manual
## Intelligent Quantitative Trading System
### Version 1.0 | Cloud PM

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Core Components](#core-components)
4. [Daily Operations](#daily-operations)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)
8. [Performance Tuning](#performance-tuning)

---

## System Overview

### Architecture
The Intelligent Quantitative Trading System consists of:
- **ML/DL Models**: LSTM with attention, XGBoost ensemble
- **RL Trading**: PPO agents with custom reward functions
- **Portfolio Management**: MPT optimization, risk parity
- **Risk Control**: Dynamic stop-loss, circuit breakers, anomaly detection
- **Data Pipeline**: Real-time collection, validation, storage
- **Dashboard**: Streamlit-based monitoring interface

### Key Features
- 4,215 stocks coverage
- 15 years historical data
- Real-time trading execution
- Advanced risk management
- Paper trading mode
- Performance analytics

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- 16GB RAM minimum
- 100GB disk space
- Internet connection for data feeds

### Installation

```bash
# 1. Clone repository
git clone [repository-url]
cd QuantProject

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your API credentials
```

### Initial Setup

```bash
# 1. Initialize database
python scripts/init_database.py

# 2. Download historical data
python scripts/download/start_full_download.py

# 3. Train models (optional - pre-trained models included)
python scripts/train_all_models.py

# 4. Verify installation
python scripts/verify_installation.py
```

---

## Core Components

### 1. Trading Engine

**Start Trading (Paper Mode)**
```bash
python main_trading.py --mode paper --capital 100000
```

**Start Trading (Production)**
```bash
python main_trading.py --mode production --capital 100000
```

**Configuration Options**
```python
# config/trading_config.json
{
    "mode": "paper",
    "initial_capital": 100000,
    "max_positions": 20,
    "position_size": 0.05,
    "stop_loss": 0.02,
    "take_profit": 0.05
}
```

### 2. ML/DL Models

**LSTM Prediction**
```python
from quantproject.models.ml_models import LSTMPricePredictor

model = LSTMPricePredictor(
    input_dim=20,
    hidden_dim=128,
    num_layers=3
)
predictions = model.predict(data)
```

**XGBoost Ensemble**
```python
from quantproject.models.ml_models import XGBoostPredictor

ensemble = XGBoostPredictor()
ensemble.train(X_train, y_train)
signals = ensemble.predict(X_test)
```

### 3. Portfolio Optimization

**MPT Optimization**
```python
from quantproject.portfolio.mpt_optimizer import MPTOptimizer

optimizer = MPTOptimizer()
weights = optimizer.optimize(
    returns_matrix,
    risk_tolerance=0.5
)
```

### 4. Risk Management

**Dynamic Stop Loss**
```python
from quantproject.risk.dynamic_stop_loss import DynamicStopLoss

stop_loss = DynamicStopLoss(
    atr_multiplier=2.0,
    trailing_percent=0.02
)
stop_price = stop_loss.calculate_stop('AAPL', current_price)
```

**Circuit Breaker**
```python
from quantproject.risk.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(initial_value=100000)
breaker.check_trigger(portfolio_value)
```

### 5. Data Collection

**Real-time Data**
```python
from quantproject.data.realtime_collector import RealtimeDataCollector

collector = RealtimeDataCollector(['AAPL', 'GOOGL'])
await collector.start_collection()
```

---

## Daily Operations

### Morning Routine (9:00 AM)

1. **System Health Check**
```bash
python scripts/morning_check.py
```

2. **Update Market Data**
```bash
python scripts/update_market_data.py
```

3. **Review Risk Limits**
```bash
python scripts/check_risk_limits.py
```

4. **Start Trading Systems**
```bash
python scripts/start_trading.py
```

### During Trading Hours

1. **Monitor Dashboard**
```bash
streamlit run dashboard/main_dashboard.py
```

2. **Check Alerts**
- View dashboard alert panel
- Check logs: `logs/alerts.log`

3. **Performance Monitoring**
- Real-time P&L tracking
- Position monitoring
- Risk metrics review

### End of Day (4:00 PM)

1. **Generate Daily Report**
```bash
python scripts/generate_daily_report.py
```

2. **Backup Data**
```bash
python scripts/backup_data.py
```

3. **Shutdown Systems**
```bash
python scripts/shutdown_trading.py
```

---

## Monitoring & Maintenance

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| System Uptime | >99.9% | <99% |
| API Response Time | <200ms | >500ms |
| Memory Usage | <80% | >90% |
| Error Rate | <1% | >5% |
| Data Lag | <100ms | >1000ms |

### Log Files

```
logs/
├── trading.log       # Trading activity
├── errors.log       # System errors
├── performance.log  # Performance metrics
├── risk.log        # Risk events
└── audit.log       # Audit trail
```

### Maintenance Tasks

**Weekly**
- Review performance reports
- Update model parameters
- Clean old log files
- Verify backup integrity

**Monthly**
- Retrain ML models
- Optimize database
- Review risk parameters
- System performance audit

---

## Troubleshooting

### Common Issues

#### 1. Data Feed Connection Lost
```bash
# Check connection
python scripts/check_data_feed.py

# Restart collector
python scripts/restart_collector.py
```

#### 2. High Memory Usage
```bash
# Clear cache
python scripts/clear_cache.py

# Restart services
python scripts/restart_services.py
```

#### 3. Model Prediction Errors
```bash
# Validate model
python scripts/validate_models.py

# Reload model
python scripts/reload_models.py
```

#### 4. Trading Execution Failed
```bash
# Check API status
python scripts/check_api_status.py

# Review error logs
tail -n 100 logs/errors.log
```

### Emergency Procedures

**Emergency Stop**
```bash
python scripts/emergency_stop.py
```

**Data Recovery**
```bash
python scripts/recover_data.py --date 2025-08-10
```

**System Rollback**
```bash
python scripts/rollback.py --version 1.0
```

---

## API Reference

### Trading API

```python
# Execute trade
POST /api/trade
{
    "symbol": "AAPL",
    "quantity": 100,
    "side": "BUY",
    "order_type": "MARKET"
}

# Get positions
GET /api/positions

# Get account info
GET /api/account
```

### Data API

```python
# Get real-time quote
GET /api/quote/{symbol}

# Get historical data
GET /api/history/{symbol}?period=1Y

# Get market status
GET /api/market/status
```

### Risk API

```python
# Get risk metrics
GET /api/risk/metrics

# Set stop loss
POST /api/risk/stop-loss
{
    "symbol": "AAPL",
    "stop_price": 145.50
}

# Check circuit breaker
GET /api/risk/circuit-breaker
```

---

## Performance Tuning

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_symbol_date ON market_data(symbol, date);
CREATE INDEX idx_portfolio_date ON portfolio_history(date);

-- Vacuum database
VACUUM;
ANALYZE;
```

### Model Optimization

```python
# Reduce model complexity
model.reduce_parameters(factor=0.8)

# Enable GPU acceleration
model.enable_gpu()

# Batch processing
model.set_batch_size(64)
```

### System Configuration

```yaml
# config/performance.yaml
threading:
  max_workers: 8
  queue_size: 1000

caching:
  enabled: true
  ttl: 3600
  max_size: 1GB

database:
  connection_pool: 20
  timeout: 30
```

---

## Best Practices

### Trading
1. Always use stop-loss orders
2. Diversify portfolio (max 5% per position)
3. Monitor risk metrics continuously
4. Review daily performance reports
5. Keep audit trail of all trades

### System Management
1. Regular backups (daily)
2. Monitor system resources
3. Keep logs for 30 days minimum
4. Test updates in paper mode first
5. Document all configuration changes

### Security
1. Never hardcode credentials
2. Use encrypted connections
3. Rotate API keys monthly
4. Enable two-factor authentication
5. Regular security audits

---

## Support

### Documentation
- User Manual: `docs/user_manual.md`
- API Docs: `docs/api_reference.md`
- FAQ: `docs/faq.md`

### Contact
- Technical Support: tech-support@quanttrading.com
- Bug Reports: GitHub Issues
- Feature Requests: GitHub Discussions

### Resources
- Training Videos: [YouTube Channel]
- Community Forum: [Discord Server]
- Knowledge Base: [Wiki]

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-10  
**Author**: Cloud PM

---

## Appendix

### A. Configuration Files
- `config/trading_config.json`
- `config/risk_config.yaml`
- `config/data_config.ini`

### B. Script Reference
- See `scripts/README.md` for complete script documentation

### C. Error Codes
- See `docs/error_codes.md` for complete error reference
