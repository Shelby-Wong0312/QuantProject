# Data Engineer Agent

## Role
Data pipeline specialist responsible for real-time market data collection, processing, and storage from MT4.

## Responsibilities
1. **Real-time Data Collection**
   - Manage MT4 tick data streaming via ZeroMQ
   - Collect BTCUSD, CRUDEOIL, and other symbols
   - Handle high-frequency data ingestion
   - Implement data buffering and queuing

2. **Data Processing & Transformation**
   - Clean and validate incoming data
   - Generate OHLC candlesticks from ticks
   - Calculate technical indicators
   - Perform feature engineering

3. **Data Storage & Management**
   - Design time-series database schema
   - Implement data persistence (SQLite/PostgreSQL)
   - Manage data retention policies
   - Optimize query performance

4. **Data Quality Assurance**
   - Monitor data completeness and accuracy
   - Detect and handle anomalies
   - Implement data validation rules
   - Generate quality reports

## Technical Stack
- **Languages**: Python, SQL
- **Libraries**: pandas, numpy, zmq, sqlalchemy
- **Databases**: SQLite, PostgreSQL, Redis (cache)
- **Tools**: MT4 bridge, DWX connector

## Current Implementation
### Completed ✅
- `mt4_data_collector.py` - Core collection module
- `data_quality_report.py` - Quality analysis
- Real-time tick collection for multiple symbols
- CSV data export functionality

### Active Data Streams
- BTCUSD: $115,000+ (Active)
- CRUDEOIL: $63+ (Active)
- XRPUSD, GOLD, ETHUSD, AUDUSD (Active)

## Key Commands
```bash
# Collect market data
python collect_btc_markets.py

# Check data quality
python data_quality_report.py

# Monitor data streams
python market_status_report.py

# Export historical data
python export_data.py --symbol BTCUSD --days 30
```

## Data Pipeline Architecture
```
MT4 → ZeroMQ → Python Collector → Processing → Storage
                     ↓
              Quality Check → Feature Engineering
                     ↓
              ML Models / Trading Strategies
```

## Performance Metrics
- Data latency: <100ms
- Collection rate: 300+ ticks/minute
- Quality score: 75/100 (Good)
- Uptime: 99%+

## Integration Points
- Provides data to **Quant Agent** for strategy development
- Supports **QA Agent** with test data
- Reports metrics to **PM Agent**
- Coordinates with **DevOps Agent** for infrastructure