"""
Main backtesting engine
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import logging
from tqdm import tqdm

from .portfolio import Portfolio
from .strategy_base import Strategy, Signal
from .performance import PerformanceAnalyzer
from .models import CombinedCostModel, TransactionCostModel, SlippageModel


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        cost_model: Optional[CombinedCostModel] = None,
        data_frequency: str = 'daily',
        enable_shorting: bool = True,
        enable_fractional_shares: bool = True
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            cost_model: Combined cost model for transaction costs and slippage
            data_frequency: Data frequency ('minute', 'hourly', 'daily')
            enable_shorting: Whether to allow short positions
            enable_fractional_shares: Whether to allow fractional shares
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CombinedCostModel()
        self.data_frequency = data_frequency
        self.enable_shorting = enable_shorting
        self.enable_fractional_shares = enable_fractional_shares
        
        # Components
        self.portfolio = Portfolio(initial_capital)
        self.performance_analyzer = PerformanceAnalyzer(
            periods_per_year=self._get_periods_per_year()
        )
        
        # State
        self.market_data = {}
        self.current_prices = {}
        self.current_datetime = None
        self.is_running = False
        
        # Results
        self.results = {}
        
    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on data frequency"""
        frequency_map = {
            'minute': 252 * 6.5 * 60,  # Trading minutes per year
            'hourly': 252 * 6.5,       # Trading hours per year
            'daily': 252,              # Trading days per year
            'weekly': 52,              # Weeks per year
            'monthly': 12              # Months per year
        }
        return frequency_map.get(self.data_frequency, 252)
    
    def add_data(
        self,
        data: pd.DataFrame,
        instrument: str,
        price_column: str = 'close'
    ) -> None:
        """
        Add market data for an instrument
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            instrument: Instrument identifier
            price_column: Column to use for execution price
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Ensure data is sorted
        data = data.sort_index()
        
        # Store data
        self.market_data[instrument] = {
            'data': data,
            'price_column': price_column
        }
        
        logger.info(f"Added data for {instrument}: {len(data)} bars")
    
    def run(
        self,
        strategy: Strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark: Optional[str] = None,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest with given strategy
        
        Args:
            strategy: Strategy instance to test
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark: Benchmark instrument for comparison
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary of backtest results
        """
        if not self.market_data:
            raise ValueError("No market data added. Use add_data() first.")
        
        # Reset state
        self._reset()
        self.is_running = True
        
        # Get date range
        all_dates = self._get_all_dates()
        if start_date:
            all_dates = all_dates[all_dates >= start_date]
        if end_date:
            all_dates = all_dates[all_dates <= end_date]
        
        if len(all_dates) == 0:
            raise ValueError("No data in specified date range")
        
        logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
        
        # Initialize strategy
        init_data = self._get_market_snapshot(all_dates[0])
        strategy.initialize(init_data)
        
        # Main backtest loop
        date_iterator = tqdm(all_dates) if progress_bar else all_dates
        
        for current_date in date_iterator:
            self.current_datetime = current_date
            
            # Update market prices
            self._update_market_prices(current_date)
            
            # Update portfolio with current prices
            self.portfolio.update_market_prices(
                self.current_prices,
                current_date
            )
            
            # Get market data snapshot
            market_snapshot = self._get_market_snapshot(current_date)
            
            # Generate signals
            portfolio_state = self.portfolio.get_portfolio_state()
            signals = strategy.generate_signals(market_snapshot, portfolio_state)
            
            # Apply risk filters
            filtered_signals = strategy.apply_risk_filters(signals, portfolio_state)
            
            # Execute signals
            self._execute_signals(filtered_signals, strategy)
            
            # Check stop losses and take profits
            self._check_exit_conditions()
        
        # Final portfolio update
        self.portfolio.update_market_prices(self.current_prices, all_dates[-1])
        
        # Analyze performance
        portfolio_history = self.portfolio.get_portfolio_history()
        transaction_history = self.portfolio.get_transaction_history()
        
        # Get benchmark returns if specified
        benchmark_returns = None
        if benchmark and benchmark in self.market_data:
            benchmark_data = self.market_data[benchmark]['data']
            benchmark_column = self.market_data[benchmark]['price_column']
            benchmark_returns = benchmark_data[benchmark_column].pct_change()
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.analyze(
            portfolio_history,
            transaction_history,
            benchmark_returns
        )
        
        # Compile results
        self.results = {
            'performance_metrics': performance_metrics,
            'portfolio_history': portfolio_history,
            'transaction_history': transaction_history,
            'strategy_summary': strategy.get_performance_summary(),
            'final_portfolio_state': self.portfolio.get_portfolio_state()
        }
        
        self.is_running = False
        
        # Print summary
        logger.info("\n" + self.performance_analyzer.get_summary_string())
        
        return self.results
    
    def _reset(self) -> None:
        """Reset engine state"""
        self.portfolio.reset()
        self.current_prices = {}
        self.current_datetime = None
        self.results = {}
    
    def _get_all_dates(self) -> pd.DatetimeIndex:
        """Get all unique dates from market data"""
        all_dates = []
        
        for instrument_data in self.market_data.values():
            all_dates.extend(instrument_data['data'].index)
        
        # Get unique sorted dates
        unique_dates = pd.DatetimeIndex(sorted(set(all_dates)))
        
        return unique_dates
    
    def _update_market_prices(self, current_date: datetime) -> None:
        """Update current market prices"""
        self.current_prices = {}
        
        for instrument, data_info in self.market_data.items():
            data = data_info['data']
            price_column = data_info['price_column']
            
            # Get price for current date
            if current_date in data.index:
                self.current_prices[instrument] = data.loc[current_date, price_column]
            else:
                # Forward fill if no data for this date
                earlier_data = data[data.index <= current_date]
                if not earlier_data.empty:
                    self.current_prices[instrument] = earlier_data[price_column].iloc[-1]
    
    def _get_market_snapshot(self, current_date: datetime) -> pd.DataFrame:
        """Get market data snapshot for current date"""
        snapshot_data = {}
        
        for instrument, data_info in self.market_data.items():
            data = data_info['data']
            
            # Get data up to current date
            historical_data = data[data.index <= current_date]
            
            if not historical_data.empty:
                # Include latest row and some history
                lookback = min(100, len(historical_data))
                snapshot_data[instrument] = historical_data.tail(lookback)
        
        # Combine into single DataFrame if single instrument
        if len(snapshot_data) == 1:
            return list(snapshot_data.values())[0]
        
        return snapshot_data
    
    def _execute_signals(self, signals: List[Signal], strategy: Strategy) -> None:
        """Execute trading signals"""
        for signal in signals:
            if signal.instrument not in self.current_prices:
                logger.warning(f"No price data for {signal.instrument}")
                continue
            
            current_price = self.current_prices[signal.instrument]
            
            # Calculate position size
            portfolio_value = self.portfolio.total_value
            current_position = self.portfolio.get_position_size(signal.instrument)
            
            # Determine target position
            if signal.target_weight is not None:
                # Weight-based sizing
                target_value = portfolio_value * signal.target_weight
                target_position = target_value / current_price
            else:
                # Use strategy's position sizing
                if signal.direction == 'BUY':
                    target_position = strategy.calculate_position_size(
                        signal, current_price, portfolio_value
                    )
                elif signal.direction == 'SELL':
                    target_position = -strategy.calculate_position_size(
                        signal, current_price, portfolio_value
                    )
                else:  # HOLD
                    target_position = current_position
            
            # Calculate order size
            order_size = target_position - current_position
            
            # Skip if order too small
            if abs(order_size) < 0.0001:
                continue
            
            # Check shorting constraints
            if not self.enable_shorting and target_position < 0:
                logger.debug(f"Shorting disabled, skipping sell signal for {signal.instrument}")
                continue
            
            # Round to whole shares if needed
            if not self.enable_fractional_shares:
                order_size = int(order_size)
                if order_size == 0:
                    continue
            
            # Calculate execution costs
            costs = self.cost_model.calculate_total_cost(
                order_size=order_size,
                current_price=current_price,
                instrument=signal.instrument,
                volume=self._get_current_volume(signal.instrument),
                volatility=self._get_volatility(signal.instrument),
                spread=self._get_spread(signal.instrument)
            )
            
            # Execute order
            success = self.portfolio.execute_order(
                instrument=signal.instrument,
                quantity=order_size,
                price=current_price,
                commission=costs['transaction_cost'],
                slippage=costs['slippage_per_unit'],
                metadata={
                    'signal': signal,
                    'strategy': strategy.name,
                    'signal_strength': signal.strength
                }
            )
            
            if success:
                logger.debug(
                    f"Executed {signal.direction} order: "
                    f"{order_size:.2f} {signal.instrument} @ {current_price:.2f}"
                )
    
    def _check_exit_conditions(self) -> None:
        """Check stop loss and take profit conditions"""
        positions_to_close = []
        
        for instrument, position in self.portfolio.positions.items():
            if instrument not in self.current_prices:
                continue
            
            current_price = self.current_prices[instrument]
            
            # Check stop loss
            if 'signal' in position.metadata:
                signal = position.metadata['signal']
                
                if signal.stop_loss and position.quantity > 0:
                    if current_price <= signal.stop_loss:
                        positions_to_close.append((instrument, "Stop Loss"))
                
                elif signal.stop_loss and position.quantity < 0:
                    if current_price >= signal.stop_loss:
                        positions_to_close.append((instrument, "Stop Loss"))
                
                # Check take profit
                if signal.take_profit and position.quantity > 0:
                    if current_price >= signal.take_profit:
                        positions_to_close.append((instrument, "Take Profit"))
                
                elif signal.take_profit and position.quantity < 0:
                    if current_price <= signal.take_profit:
                        positions_to_close.append((instrument, "Take Profit"))
        
        # Close positions
        for instrument, reason in positions_to_close:
            position = self.portfolio.get_position(instrument)
            if position:
                current_price = self.current_prices[instrument]
                
                # Calculate costs
                costs = self.cost_model.calculate_total_cost(
                    order_size=-position.quantity,
                    current_price=current_price,
                    instrument=instrument
                )
                
                # Execute closing order
                self.portfolio.execute_order(
                    instrument=instrument,
                    quantity=-position.quantity,
                    price=current_price,
                    commission=costs['transaction_cost'],
                    slippage=costs['slippage_per_unit'],
                    metadata={'reason': reason}
                )
                
                logger.debug(f"Closed position in {instrument}: {reason}")
    
    def _get_current_volume(self, instrument: str) -> float:
        """Get current volume for instrument"""
        if instrument not in self.market_data:
            return 1000000  # Default volume
        
        data = self.market_data[instrument]['data']
        if self.current_datetime in data.index and 'volume' in data.columns:
            return data.loc[self.current_datetime, 'volume']
        
        return 1000000
    
    def _get_volatility(self, instrument: str) -> float:
        """Get volatility estimate for instrument"""
        if instrument not in self.market_data:
            return 0.02  # Default 2% volatility
        
        data = self.market_data[instrument]['data']
        price_column = self.market_data[instrument]['price_column']
        
        # Calculate rolling volatility
        returns = data[price_column].pct_change()
        volatility = returns.rolling(window=20).std()
        
        if self.current_datetime in volatility.index:
            return volatility.loc[self.current_datetime]
        
        return 0.02
    
    def _get_spread(self, instrument: str) -> Optional[float]:
        """Get bid-ask spread for instrument"""
        if instrument not in self.market_data:
            return None
        
        data = self.market_data[instrument]['data']
        
        # Check if we have bid/ask data
        if 'bid' in data.columns and 'ask' in data.columns:
            if self.current_datetime in data.index:
                bid = data.loc[self.current_datetime, 'bid']
                ask = data.loc[self.current_datetime, 'ask']
                return ask - bid
        
        return None
    
    def get_results_summary(self) -> str:
        """Get formatted results summary"""
        if not self.results:
            return "No results available. Run backtest first."
        
        return self.performance_analyzer.get_summary_string()
    
    def plot_results(self, show_positions: bool = True) -> None:
        """Plot backtest results (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                logger.error("No results to plot. Run backtest first.")
                return
            
            portfolio_history = self.results['portfolio_history']
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Portfolio value
            ax1 = axes[0]
            ax1.plot(portfolio_history.index, portfolio_history['total_value'])
            ax1.set_ylabel('Portfolio Value')
            ax1.set_title('Backtest Results')
            ax1.grid(True, alpha=0.3)
            
            # Returns
            ax2 = axes[1]
            returns = portfolio_history['total_value'].pct_change()
            cumulative_returns = (1 + returns).cumprod() - 1
            ax2.plot(portfolio_history.index, cumulative_returns * 100)
            ax2.set_ylabel('Cumulative Return (%)')
            ax2.grid(True, alpha=0.3)
            
            # Drawdown
            ax3 = axes[2]
            running_max = portfolio_history['total_value'].expanding().max()
            drawdown = (portfolio_history['total_value'] - running_max) / running_max
            ax3.fill_between(portfolio_history.index, drawdown * 100, alpha=0.3, color='red')
            ax3.set_ylabel('Drawdown (%)')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot results.")