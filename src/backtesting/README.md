# Backtesting Framework

A comprehensive, event-driven backtesting framework for quantitative trading strategies with realistic transaction cost modeling and detailed performance analysis.

## Features

### Core Components

1. **BacktestEngine** (`engine.py`)
   - Event-driven architecture for realistic simulation
   - Support for multiple instruments and data frequencies
   - Automatic stop-loss and take-profit handling
   - Configurable transaction costs and slippage models

2. **Portfolio Management** (`portfolio.py`)
   - Real-time position tracking
   - P&L calculation (realized and unrealized)
   - Transaction history recording
   - Portfolio state snapshots

3. **Strategy Framework** (`strategy_base.py`)
   - Abstract base class for strategy development
   - Signal generation interface
   - Risk management filters
   - Position sizing methods (fixed, volatility-based, Kelly criterion)

4. **Performance Analysis** (`performance.py`)
   - Comprehensive metrics calculation
   - Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
   - Drawdown analysis
   - Trading statistics (win rate, profit factor)
   - Benchmark comparison

5. **Cost Models** (`models.py`)
   - Transaction cost models (fixed, percentage, tiered)
   - Slippage models (fixed, linear, square-root)
   - Combined cost calculation

## Quick Start

### Basic Usage

```python
from src.backtesting import BacktestEngine, BuyAndHoldStrategy

# Create engine
engine = BacktestEngine(
    initial_capital=100000,
    data_frequency='daily'
)

# Add market data
engine.add_data(price_data, 'AAPL')

# Run strategy
strategy = BuyAndHoldStrategy(['AAPL'])
results = engine.run(strategy)

# View results
print(engine.get_results_summary())
```

### Custom Strategy Development

```python
from src.backtesting import Strategy, Signal

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__(name="MyStrategy")
    
    def initialize(self, market_data):
        # Initialize strategy state
        pass
    
    def generate_signals(self, market_data, portfolio_state):
        signals = []
        
        # Your strategy logic here
        if condition_met:
            signal = Signal(
                instrument='AAPL',
                direction='BUY',
                strength=1.0
            )
            signals.append(signal)
        
        return signals
```

### Advanced Configuration

```python
from src.backtesting.models import PercentageTransactionCost, SquareRootSlippage, CombinedCostModel

# Configure realistic costs
cost_model = CombinedCostModel(
    transaction_cost_model=PercentageTransactionCost(0.001),  # 0.1%
    slippage_model=SquareRootSlippage(
        temporary_impact=0.1,
        permanent_impact=0.05
    )
)

# Create engine with custom settings
engine = BacktestEngine(
    initial_capital=1000000,
    cost_model=cost_model,
    data_frequency='minute',
    enable_shorting=True,
    enable_fractional_shares=True
)
```

## Strategy Development Guide

### 1. Risk Parameters

```python
strategy.risk_parameters = {
    'max_position_size': 0.1,      # Max 10% per position
    'max_portfolio_exposure': 1.0,  # Max 100% invested
    'stop_loss_pct': 0.02,         # 2% stop loss
    'position_sizing_method': 'volatility',  # or 'fixed', 'kelly'
    'max_positions': 10,           # Maximum concurrent positions
}
```

### 2. Signal Generation

```python
signal = Signal(
    instrument='AAPL',
    direction='BUY',  # or 'SELL', 'HOLD'
    strength=0.8,     # Signal confidence (0-1)
    target_weight=0.05,  # Optional: target portfolio weight
    stop_loss=95.0,      # Optional: stop loss price
    take_profit=105.0,   # Optional: take profit price
    metadata={'reason': 'breakout'}  # Custom metadata
)
```

### 3. Position Sizing Methods

- **Fixed**: Allocate fixed percentage of portfolio
- **Volatility**: Size inversely proportional to volatility
- **Kelly Criterion**: Optimal sizing based on win probability

## Performance Metrics

The framework calculates 30+ performance metrics including:

### Returns
- Total return
- Annualized return
- Volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio

### Risk
- Maximum drawdown
- Value at Risk (VaR)
- Conditional VaR
- Downside deviation
- Skewness & Kurtosis

### Trading
- Win rate
- Profit factor
- Average win/loss ratio
- Trade frequency
- Commission & slippage costs

### Benchmark Comparison
- Alpha & Beta
- Information ratio
- Tracking error
- Correlation

## Example Strategies

### Moving Average Crossover
```python
class SMAStrategy(Strategy):
    def generate_signals(self, data, portfolio):
        fast_sma = data['close'].rolling(20).mean()
        slow_sma = data['close'].rolling(50).mean()
        
        if fast_sma.iloc[-1] > slow_sma.iloc[-1]:
            return [Signal('AAPL', 'BUY', 1.0)]
        else:
            return [Signal('AAPL', 'SELL', 1.0)]
```

### Mean Reversion
```python
class MeanReversionStrategy(Strategy):
    def generate_signals(self, data, portfolio):
        price = data['close'].iloc[-1]
        mean = data['close'].rolling(20).mean().iloc[-1]
        std = data['close'].rolling(20).std().iloc[-1]
        
        if price < mean - 2*std:
            return [Signal('AAPL', 'BUY', 1.0)]
        elif price > mean + 2*std:
            return [Signal('AAPL', 'SELL', 1.0)]
        
        return []
```

## Data Requirements

Market data should be a pandas DataFrame with:
- **Index**: DatetimeIndex
- **Required columns**: At minimum 'close' prices
- **Optional columns**: 'open', 'high', 'low', 'volume', 'bid', 'ask'

```python
# Example data structure
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.date_range('2023-01-01', '2023-12-31', freq='D'))
```

## Best Practices

1. **Always validate data** before backtesting
2. **Include realistic costs** - both commission and slippage
3. **Test multiple market conditions** - bull, bear, and sideways markets
4. **Monitor drawdowns** - not just returns
5. **Use appropriate position sizing** - avoid over-leveraging
6. **Implement proper risk management** - stop losses, position limits

## Visualization

The framework includes built-in plotting:

```python
# Plot results
engine.plot_results(show_positions=True)
```

This displays:
- Portfolio value over time
- Cumulative returns
- Drawdown periods
- Trade markers (optional)

## Performance Tips

1. **Vectorize calculations** in strategy logic
2. **Pre-calculate indicators** where possible
3. **Use appropriate data frequency** - higher frequency = slower backtest
4. **Limit history lookback** to necessary periods
5. **Disable progress bar** for automated testing

## Next Steps

This backtesting framework provides the foundation for:
- Strategy development and testing
- Parameter optimization
- Walk-forward analysis
- Monte Carlo simulations
- Live trading integration

The modular design allows easy extension with:
- Custom cost models
- New performance metrics
- Alternative position sizing methods
- Advanced order types