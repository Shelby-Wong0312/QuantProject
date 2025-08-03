# Capital.com Data Provider

This module provides integration with Capital.com's REST API for fetching market data and managing trades.

## Features

- **Authentication**: Secure API authentication with token management
- **Market Data**: Search markets, get market details, fetch historical prices
- **Trading**: Create positions, close positions, view open positions
- **Account Management**: Access account information and balances

## Setup

1. **Get API Credentials**:
   - Sign up for a Capital.com account (demo or live)
   - Navigate to the API section in your account settings
   - Generate your API key

2. **Configure Environment Variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your credentials
   CAPITAL_COM_API_KEY=your_api_key
   CAPITAL_COM_IDENTIFIER=your_identifier
   CAPITAL_COM_PASSWORD=your_password
   CAPITAL_COM_DEMO_MODE=True  # Set to False for live trading
   ```

3. **Install Dependencies**:
   ```bash
   pip install requests
   ```

## Usage

### Basic Usage

```python
from src.data_providers.capital_com.api_client import CapitalComClient

# Create client and connect
client = CapitalComClient()
if client.connect():
    # Search for markets
    markets = client.search_markets("AAPL")
    
    # Get historical data
    prices = client.get_historical_prices(
        epic="AAPL",
        resolution="HOUR",
        max_bars=100
    )
    
    # Disconnect when done
    client.disconnect()
```

### Using Context Manager

```python
from src.data_providers.capital_com.api_client import CapitalComClient

# Automatic connection and disconnection
with CapitalComClient() as client:
    accounts = client.get_accounts()
    positions = client.get_positions()
```

### Available Resolutions

- `MINUTE` - 1 minute bars
- `MINUTE_5` - 5 minute bars
- `MINUTE_15` - 15 minute bars
- `MINUTE_30` - 30 minute bars
- `HOUR` - 1 hour bars
- `HOUR_4` - 4 hour bars
- `DAY` - Daily bars
- `WEEK` - Weekly bars

## API Methods

### Authentication
- `connect()` - Authenticate with API
- `disconnect()` - Logout from API
- `is_authenticated()` - Check authentication status

### Market Data
- `search_markets(search_term, limit)` - Search for markets
- `get_market_details(epic)` - Get detailed market information
- `get_historical_prices(epic, resolution, max_bars, from_date, to_date)` - Get historical price data

### Trading
- `get_positions()` - Get all open positions
- `create_position(epic, direction, size, stop_level, profit_level)` - Open a new position
- `close_position(deal_id)` - Close an existing position

### Account
- `get_accounts()` - Get account information

## Error Handling

The client includes comprehensive error handling:
- Network errors are caught and logged
- Authentication errors trigger re-authentication attempts
- API rate limits are respected with retry logic

## Security Notes

- Never commit your `.env` file with actual credentials
- Use demo mode for testing and development
- Store production credentials securely
- The client automatically manages authentication tokens

## Next Steps

This module provides the foundation for the data acquisition layer. Next implementations should include:
- Data caching mechanism
- Real-time data streaming
- Advanced error recovery
- Data persistence layer