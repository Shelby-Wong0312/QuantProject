"""
Backtest Engine - Core backtesting framework
Provides event-driven backtesting with realistic transaction costs and performance analysis
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from .portfolio import Portfolio
from .performance import PerformanceAnalyzer
from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration"""

    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    benchmark: str = "SPY"
    rebalance_frequency: str = "daily"  # daily, weekly, monthly


class BacktestEngine:
    """
    Core backtesting engine for trading strategies

    Features:
    - Event-driven backtesting
    - Realistic transaction costs
    - Performance analytics
    - Risk management
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.portfolio = Portfolio(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
        )
        self.performance_analyzer = PerformanceAnalyzer()

        # Market data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_date: Optional[datetime] = None

        # Backtest state
        self.is_running = False
        self.current_bar = 0
        self.total_bars = 0

        # Results storage
        self.results: Dict[str, Any] = {}
        self.daily_values: List[float] = []
        self.daily_returns: List[float] = []
        self.trade_log: List[Dict] = []

        logger.info(
            f"BacktestEngine initialized with ${config.initial_capital:,.2f} capital"
        )

    def add_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Add market data for a symbol

        Args:
            data: OHLCV data with datetime index
            symbol: Symbol name
        """
        # Validate data format
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            # Create missing columns with close price
            for col in missing_columns:
                if col == "volume":
                    data[col] = 1000000  # Default volume
                else:
                    data[col] = data["close"]

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Sort by date
        data.sort_index()

        # Calculate returns
        data["returns"] = data["close"].pct_change().fillna(0)

        self.market_data[symbol] = data
        logger.info(
            f"Added data for {symbol}: {len(data)} bars from {data.index[0]} to {data.index[-1]}"
        )

    def run_backtest(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """
        Run complete backtest

        Args:
            strategy: Trading strategy to test

        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")
        self.is_running = True

        # Get date range
        all_dates = self._get_date_range()
        if not all_dates:
            raise ValueError("No market data available")

        self.total_bars = len(all_dates)
        strategy.reset()  # Reset strategy state

        # Initialize tracking
        portfolio_values = []
        returns = []
        positions_history = []

        try:
            # Main backtest loop
            for i, date in enumerate(all_dates):
                self.current_date = date
                self.current_bar = i

                # Get market data for current date
                current_data = self._get_current_market_data(date)
                if not current_data:
                    portfolio_values.append(self.portfolio.total_value)
                    returns.append(0)
                    continue

                # Update portfolio with current prices
                self._update_portfolio_prices(current_data)

                # Check rebalancing
                if self._should_rebalance(date, i):
                    # Generate signals
                    self._generate_signals(strategy, current_data, date)

                    # Execute trades
                    trades = self._execute_signals(signals, current_data)

                    # Log trades
                    for trade in trades:
                        trade["date"] = date
                        self.trade_log.append(trade)

                # Record portfolio value
                portfolio_value = self.portfolio.calculate_total_value(current_data)
                portfolio_values.append(portfolio_value)

                # Calculate daily return
                if i > 0:
                    daily_return = (
                        portfolio_value - portfolio_values[i - 1]
                    ) / portfolio_values[i - 1]
                    returns.append(daily_return)
                else:
                    returns.append(0)

                # Record positions
                positions_history.append(
                    {
                        "date": date,
                        "positions": self.portfolio.get_positions_summary(),
                        "total_value": portfolio_value,
                        "cash": self.portfolio.cash,
                    }
                )

                # Progress logging
                if i % 252 == 0 and i > 0:  # Every trading year
                    progress = (i / self.total_bars) * 100
                    total_return = (
                        portfolio_value - self.config.initial_capital
                    ) / self.config.initial_capital
                    logger.info(
                        f"Progress: {progress:.1f}% | Year {i//252} | "
                        f"Portfolio: ${portfolio_value:,.2f} | Return: {total_return:.2%}"
                    )

            # Calculate final results
            self.results = self._calculate_final_results(
                portfolio_values, returns, positions_history, all_dates
            )

            logger.info("Backtest completed successfully")
            return self.results

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

        finally:
            self.is_running = False

    def _get_date_range(self) -> List[datetime]:
        """Get unified date range from all market data"""
        if not self.market_data:
            return []

        # Get union of all date ranges (use available data from any symbol)
        all_dates = set()
        for symbol, data in self.market_data.items():
            symbol_dates = set(data.index)
            all_dates = all_dates.union(symbol_dates)

        if not all_dates:
            return []

        # Filter by config date range
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)

        # If no specific date range is set, use all available data
        if (
            self.config.start_date == "2020-01-01"
            and self.config.end_date == "2024-12-31"
        ):
            # Use actual data range
            all_dates_list = sorted(all_dates)
            if all_dates_list:
                start_date = all_dates_list[0]
                end_date = all_dates_list[-1]

        filtered_dates = [
            date for date in sorted(all_dates) if start_date <= date <= end_date
        ]

        return filtered_dates

    def _get_current_market_data(self, date: datetime) -> Dict[str, Dict]:
        """Get market data for current date"""
        current_data = {}

        for symbol, data in self.market_data.items():
            if date in data.index:
                row = data.loc[date]
                current_data[symbol] = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "returns": row["returns"],
                }

        return current_data

    def _update_portfolio_prices(self, market_data: Dict[str, Dict]) -> None:
        """Update portfolio with current market prices"""
        for symbol, data in market_data.items():
            self.portfolio.update_position_price(symbol, data["close"])

    def _should_rebalance(self, date: datetime, bar_index: int) -> bool:
        """Check if rebalancing should occur"""
        freq = self.config.rebalance_frequency.lower()

        if freq == "daily":
            return True
        elif freq == "weekly":
            return date.weekday() == 0  # Monday
        elif freq == "monthly":
            # First trading day of month
            if bar_index == 0:
                return True
            prev_date = date - timedelta(days=1)
            return date.month != prev_date.month

        return False

    def _generate_signals(
        self, strategy: BaseStrategy, market_data: Dict[str, Dict], date: datetime
    ) -> List:
        """Generate trading signals from strategy"""
        try:
            # Prepare historical data for strategy
            strategy_data = {}

            for symbol in market_data.keys():
                if symbol in self.market_data:
                    # Get historical data up to current date
                    historical = self.market_data[symbol].loc[:date]
                    if len(historical) >= 20:  # Minimum history needed
                        # Set symbol name in data for signal generation
                        historical.index.name = symbol
                        strategy_data[symbol] = historical

            if not strategy_data:
                return []

            # Generate signals for each symbol
            []
            for symbol, data in strategy_data.items():
                try:
                    symbol_signals = strategy.calculate_signals(data)
                    # Ensure signals have correct symbol
                    for signal in symbol_signals:
                        if hasattr(signal, "symbol"):
                            signal.symbol = symbol
                    signals.extend(symbol_signals)
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol}: {e}")
                    continue

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    def _execute_signals(
        self, signals: List, market_data: Dict[str, Dict]
    ) -> List[Dict]:
        """Execute trading signals"""
        executed_trades = []

        for signal in signals:
            try:
                symbol = signal.symbol
                if symbol not in market_data:
                    continue

                current_price = market_data[symbol]["close"]

                # Calculate position size
                portfolio_value = self.portfolio.calculate_total_value(market_data)
                position_size = self._calculate_position_size(
                    signal, portfolio_value, current_price
                )

                if abs(position_size) < 1:  # Skip tiny positions
                    continue

                # Execute trade
                trade_result = self.portfolio.execute_trade(
                    symbol=symbol,
                    quantity=position_size,
                    price=current_price,
                    signal_type=signal.signal_type.value,
                    timestamp=self.current_date,
                )

                if trade_result["success"]:
                    executed_trades.append(trade_result)
                    logger.debug(
                        f"Executed: {symbol} {signal.signal_type.value} "
                        f"{position_size:.0f} @ ${current_price:.2f}"
                    )

            except Exception as e:
                logger.error(f"Error executing signal: {e}")
                continue

        return executed_trades

    def _calculate_position_size(
        self, signal, portfolio_value: float, current_price: float
    ) -> float:
        """Calculate position size based on signal and risk management"""
        # Simple position sizing: 5% of portfolio per position
        position_value = portfolio_value * 0.05
        base_size = position_value / current_price

        # Adjust based on signal strength
        adjusted_size = base_size * signal.strength

        # Apply signal direction
        if signal.signal_type.value in ["SELL", "STRONG_SELL"]:
            adjusted_size = -adjusted_size

        return adjusted_size

    def _calculate_final_results(
        self,
        portfolio_values: List[float],
        returns: List[float],
        positions_history: List[Dict],
        dates: List[datetime],
    ) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""

        # Convert to pandas Series for easier analysis
        portfolio_series = pd.Series(portfolio_values, index=dates)

        # Ensure returns and dates alignment
        if len(returns) > 0:
            returns_dates = dates[: len(returns)]  # Use only dates that have returns
            returns_series = pd.Series(returns, index=returns_dates)
        else:
            returns_series = pd.Series([], dtype=float)

        # Use performance analyzer
        performance_metrics = self.performance_analyzer.calculate_metrics(
            portfolio_values=portfolio_series,
            returns=returns_series,
            trades=self.trade_log,
            initial_capital=self.config.initial_capital,
        )

        # Additional backtest-specific metrics
        results = {
            "config": {
                "initial_capital": self.config.initial_capital,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "commission": self.config.commission,
                "slippage": self.config.slippage,
            },
            "performance": performance_metrics,
            "portfolio_values": portfolio_series,
            "returns": returns_series,
            "trades": self.trade_log,
            "positions_history": positions_history,
            "final_portfolio": self.portfolio.get_positions_summary(),
            "total_bars": self.total_bars,
            "symbols_traded": list(set(trade["symbol"] for trade in self.trade_log)),
        }

        # Summary statistics
        final_value = portfolio_values[-1]
        total_return = (
            final_value - self.config.initial_capital
        ) / self.config.initial_capital
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        results["summary"] = {
            "total_return": total_return,
            "annual_return": annual_return,
            "final_value": final_value,
            "total_trades": len(self.trade_log),
            "years": years,
            "avg_trades_per_year": len(self.trade_log) / years if years > 0 else 0,
        }

        return results

    def get_results_summary(self) -> str:
        """Get formatted results summary"""
        if not self.results:
            return "No backtest results available. Run backtest first."

        self.results["summary"]
        self.results["performance"]

        """
BACKTEST RESULTS SUMMARY
========================
Period: {self.config.start_date} to {self.config.end_date}
Initial Capital: ${self.config.initial_capital:,.2f}
Final Value: ${summary['final_value']:,.2f}

PERFORMANCE METRICS
-------------------
Total Return: {summary['total_return']:.2%}
Annual Return: {summary['annual_return']:.2%}
Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}
Max Drawdown: {perf.get('max_drawdown', 0):.2%}
Win Rate: {perf.get('win_rate', 0):.2%}

TRADING STATISTICS
------------------
Total Trades: {summary['total_trades']}
Avg Trades/Year: {summary['avg_trades_per_year']:.1f}
Profit Factor: {perf.get('profit_factor', 0):.2f}

RISK METRICS
------------
Volatility: {perf.get('volatility', 0):.2%}
VaR (95%): {perf.get('var_95', 0):.2%}
        """

        return report.strip()

    def save_results(self, filepath: str) -> None:
        """Save backtest results to file"""
        if not self.results:
            logger.error("No results to save")
            return

        # Convert pandas objects to serializable format
        results_copy = self.results.copy()

        # Convert Series to dict
        if "portfolio_values" in results_copy:
            results_copy["portfolio_values"] = results_copy[
                "portfolio_values"
            ].to_dict()
        if "returns" in results_copy:
            results_copy["returns"] = results_copy["returns"].to_dict()

        # Save to JSON
        import json

        with open(filepath, "w") as f:
            json.dump(results_copy, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")


def run_backtest(
    strategy: BaseStrategy,
    data: Dict[str, pd.DataFrame],
    initial_capital: float = 100000,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> Dict[str, Any]:
    """
    Convenience function to run a quick backtest

    Args:
        strategy: Trading strategy
        data: Dictionary of symbol -> DataFrame
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate

    Returns:
        Backtest results
    """
    # Determine date range from data
    all_dates = []
    for df in data.values():
        all_dates.extend(df.index.tolist())

    start_date = min(all_dates).strftime("%Y-%m-%d")
    end_date = max(all_dates).strftime("%Y-%m-%d")

    # Create config
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        start_date=start_date,
        end_date=end_date,
    )

    # Create engine and run backtest
    engine = BacktestEngine(config)

    # Add data
    for symbol, df in data.items():
        engine.add_data(df, symbol)

    # Run backtest
    results = engine.run_backtest(strategy)

    return results
