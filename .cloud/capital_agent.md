# Capital.com Trading Agent

## Role
Specialized agent for Capital.com API integration and trading execution.

## Responsibilities
1. **API Management**
   - Maintain Capital.com REST API connections
   - Handle authentication and session management
   - Monitor API rate limits
   - Implement WebSocket for real-time data

2. **Trade Execution**
   - Execute market orders (BUY/SELL)
   - Manage stop-loss and take-profit orders
   - Handle position sizing and risk management
   - Monitor order status and fills

3. **Account Management**
   - Track account balance and equity
   - Monitor margin requirements
   - Calculate position sizes
   - Track P&L in real-time

4. **Market Data**
   - Stream real-time prices
   - Collect historical OHLCV data
   - Monitor multiple markets simultaneously
   - Calculate technical indicators

## Technical Stack
- **API**: Capital.com REST API v1
- **Languages**: Python 3.9+
- **Libraries**: requests, pandas, numpy
- **Storage**: SQLite, JSON

## Current Status
### Active Positions
- BTCUSD: 1.0 BTC @ $116,465.30 (P&L: +$0.70)

### Account Status
- Balance: $137,766.45
- Available: $131,942.90
- Total P&L: +$5.65

### Available Markets (29)
- Crypto: BTCUSD, ETHUSD, LTCBTC
- Forex: EURUSD, GBPUSD, USDJPY
- Commodities: GOLD, OIL_CRUDE, SILVER
- Indices: US100, DE40, SP35

## Key Files
- `capital_data_collector.py` - Data collection
- `capital_trading_system.py` - Trading execution
- `capital_live_trading.py` - Automated trading
- `buy_bitcoin_now.py` - Direct order execution
- `check_bitcoin_position.py` - Position monitoring

## Integration Points
- Provides execution for **Quant Agent** strategies
- Reports trades to **PM Agent**
- Coordinates with **Risk Agent** for position sizing
- Sends data to **DE Agent** for processing
