"""
Example usage of the backtesting framework
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from engine import BacktestEngine
from strategy_base import Strategy, Signal, BuyAndHoldStrategy
from models import PercentageTransactionCost, LinearSlippage, CombinedCostModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_data(symbol: str = 'TEST', days: int = 252) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = close_prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, days)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, days)))
    data['volume'] = np.random.randint(1000000, 5000000, days)
    
    return data


class SimpleMovingAverageCrossover(Strategy):
    """
    Simple moving average crossover strategy
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        Initialize SMA crossover strategy
        
        Args:
            fast_period: Fast SMA period
            slow_period: Slow SMA period
        """
        super().__init__(
            name="SMA_Crossover",
            parameters={'fast_period': fast_period, 'slow_period': slow_period}
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._last_signal = None
    
    def initialize(self, market_data: pd.DataFrame) -> None:
        """Initialize strategy"""
        self._initialized = True
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """Generate signals based on SMA crossover"""
        signals = []
        
        # Assume single instrument for simplicity
        if isinstance(market_data, dict):
            # Multiple instruments
            for instrument, data in market_data.items():
                signal = self._check_crossover(data, instrument)
                if signal:
                    signals.append(signal)
        else:
            # Single instrument
            signal = self._check_crossover(market_data, 'TEST')
            if signal:
                signals.append(signal)
        
        return signals
    
    def _check_crossover(self, data: pd.DataFrame, instrument: str) -> Optional[Signal]:
        """Check for SMA crossover"""
        if len(data) < self.slow_period:
            return None
        
        # Calculate SMAs
        fast_sma = data['close'].rolling(window=self.fast_period).mean()
        slow_sma = data['close'].rolling(window=self.slow_period).mean()
        
        # Get current and previous values
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2]
        prev_slow = slow_sma.iloc[-2]
        
        # Check for crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            # Golden cross - buy signal
            signal = Signal(
                instrument=instrument,
                direction='BUY',
                strength=1.0,
                metadata={
                    'fast_sma': current_fast,
                    'slow_sma': current_slow,
                    'current_price': data['close'].iloc[-1]
                }
            )
            self._last_signal = 'BUY'
            return signal
        
        elif prev_fast >= prev_slow and current_fast < current_slow:
            # Death cross - sell signal
            signal = Signal(
                instrument=instrument,
                direction='SELL',
                strength=1.0,
                metadata={
                    'fast_sma': current_fast,
                    'slow_sma': current_slow,
                    'current_price': data['close'].iloc[-1]
                }
            )
            self._last_signal = 'SELL'
            return signal
        
        return None


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands
    """
    
    def __init__(self, lookback_period: int = 20, num_std: float = 2.0):
        """
        Initialize mean reversion strategy
        
        Args:
            lookback_period: Period for moving average
            num_std: Number of standard deviations for bands
        """
        super().__init__(
            name="MeanReversion",
            parameters={
                'lookback_period': lookback_period,
                'num_std': num_std
            }
        )
        self.lookback_period = lookback_period
        self.num_std = num_std
        self._positions = {}
    
    def initialize(self, market_data: pd.DataFrame) -> None:
        """Initialize strategy"""
        self._initialized = True
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """Generate mean reversion signals"""
        signals = []
        
        if len(market_data) < self.lookback_period:
            return signals
        
        # Calculate Bollinger Bands
        close_prices = market_data['close']
        sma = close_prices.rolling(window=self.lookback_period).mean()
        std = close_prices.rolling(window=self.lookback_period).std()
        
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        current_price = close_prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Get current position
        instrument = 'TEST'
        current_position = 0
        if portfolio_state and 'positions' in portfolio_state:
            if instrument in portfolio_state['positions']:
                current_position = portfolio_state['positions'][instrument]['quantity']
        
        # Generate signals
        if current_price <= current_lower and current_position <= 0:
            # Price at lower band - buy signal
            signal = Signal(
                instrument=instrument,
                direction='BUY',
                strength=min(1.0, (sma.iloc[-1] - current_price) / (sma.iloc[-1] - current_lower)),
                metadata={
                    'current_price': current_price,
                    'lower_band': current_lower,
                    'upper_band': current_upper,
                    'sma': sma.iloc[-1]
                }
            )
            signals.append(signal)
        
        elif current_price >= current_upper and current_position >= 0:
            # Price at upper band - sell signal
            signal = Signal(
                instrument=instrument,
                direction='SELL',
                strength=min(1.0, (current_price - sma.iloc[-1]) / (current_upper - sma.iloc[-1])),
                metadata={
                    'current_price': current_price,
                    'lower_band': current_lower,
                    'upper_band': current_upper,
                    'sma': sma.iloc[-1]
                }
            )
            signals.append(signal)
        
        elif abs(current_price - sma.iloc[-1]) < std.iloc[-1] * 0.1:
            # Close to mean - close position
            if current_position != 0:
                signal = Signal(
                    instrument=instrument,
                    direction='SELL' if current_position > 0 else 'BUY',
                    strength=1.0,
                    metadata={'reason': 'mean_reversion'}
                )
                signals.append(signal)
        
        return signals


def example_basic_backtest():
    """Basic backtest example"""
    print("\n" + "="*60)
    print("BASIC BACKTEST EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=252)
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000,
        data_frequency='daily'
    )
    
    # Add data
    engine.add_data(data, 'TEST')
    
    # Create and run strategy
    strategy = BuyAndHoldStrategy(['TEST'])
    results = engine.run(strategy, progress_bar=False)
    
    print("\nBacktest completed!")
    print(f"Final portfolio value: ${results['final_portfolio_state']['total_value']:,.2f}")
    print(f"Total return: {results['performance_metrics']['total_return']*100:.2f}%")


def example_sma_crossover():
    """SMA crossover strategy example"""
    print("\n" + "="*60)
    print("SMA CROSSOVER STRATEGY EXAMPLE")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(days=500)
    
    # Set up cost model
    cost_model = CombinedCostModel(
        transaction_cost_model=PercentageTransactionCost(0.001),  # 0.1% commission
        slippage_model=LinearSlippage(0.1)  # Linear market impact
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000,
        cost_model=cost_model,
        data_frequency='daily'
    )
    
    # Add data
    engine.add_data(data, 'TEST')
    
    # Create and run strategy
    strategy = SimpleMovingAverageCrossover(fast_period=20, slow_period=50)
    results = engine.run(strategy, progress_bar=True)
    
    # Display results
    print("\n" + engine.get_results_summary())
    
    # Plot results if matplotlib is available
    try:
        engine.plot_results()
    except:
        print("Plotting not available")


def example_mean_reversion():
    """Mean reversion strategy example"""
    print("\n" + "="*60)
    print("MEAN REVERSION STRATEGY EXAMPLE")
    print("="*60)
    
    # Create sample data with more volatility
    data = create_sample_data(days=500)
    
    # Initialize backtest engine with custom risk parameters
    engine = BacktestEngine(
        initial_capital=100000,
        data_frequency='daily',
        enable_shorting=True
    )
    
    # Add data
    engine.add_data(data, 'TEST')
    
    # Create strategy with custom risk parameters
    strategy = MeanReversionStrategy(lookback_period=20, num_std=2.0)
    strategy.risk_parameters.update({
        'max_position_size': 0.2,  # 20% max per position
        'stop_loss_pct': 0.03,     # 3% stop loss
        'use_stop_loss': True
    })
    
    # Run backtest
    results = engine.run(strategy, progress_bar=True)
    
    # Display detailed metrics
    metrics = results['performance_metrics']
    print(f"\nSharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")


def example_multiple_instruments():
    """Backtest with multiple instruments"""
    print("\n" + "="*60)
    print("MULTIPLE INSTRUMENTS EXAMPLE")
    print("="*60)
    
    # Create sample data for multiple instruments
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
    # Initialize engine
    engine = BacktestEngine(initial_capital=100000)
    
    # Add data for each instrument
    for symbol in symbols:
        data = create_sample_data(symbol, days=252)
        engine.add_data(data, symbol)
    
    # Run equal-weight buy and hold
    strategy = BuyAndHoldStrategy(
        instruments=symbols,
        weights=[0.33, 0.33, 0.34]
    )
    
    results = engine.run(strategy, progress_bar=False)
    
    # Show portfolio composition
    final_state = results['final_portfolio_state']
    print("\nFinal Portfolio Composition:")
    for symbol in symbols:
        if symbol in final_state['positions']:
            position = final_state['positions'][symbol]
            print(f"  {symbol}: {position['quantity']:.2f} shares, "
                  f"Value: ${position['market_value']:,.2f}")


if __name__ == "__main__":
    # Run all examples
    example_basic_backtest()
    example_sma_crossover()
    example_mean_reversion()
    example_multiple_instruments()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)