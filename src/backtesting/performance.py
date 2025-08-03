"""
Performance analysis for backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats


logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Calculate performance metrics for backtesting results
    """
    
    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Initialize performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio
            periods_per_year: Number of trading periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._results = {}
        
    def analyze(
        self,
        portfolio_history: pd.DataFrame,
        transaction_history: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis
        
        Args:
            portfolio_history: Portfolio value history
            transaction_history: Transaction history
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        if portfolio_history.empty:
            logger.warning("Empty portfolio history")
            return {}
        
        # Calculate returns
        returns = self._calculate_returns(portfolio_history)
        
        # Basic metrics
        metrics = {
            'total_return': self._total_return(portfolio_history),
            'annualized_return': self._annualized_return(returns),
            'volatility': self._volatility(returns),
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'max_drawdown': self._max_drawdown(portfolio_history),
            'calmar_ratio': self._calmar_ratio(returns, portfolio_history),
            'win_rate': self._win_rate(transaction_history),
            'profit_factor': self._profit_factor(transaction_history),
            'average_win_loss_ratio': self._average_win_loss_ratio(transaction_history)
        }
        
        # Risk metrics
        risk_metrics = {
            'value_at_risk_95': self._value_at_risk(returns, 0.95),
            'value_at_risk_99': self._value_at_risk(returns, 0.99),
            'conditional_value_at_risk_95': self._conditional_value_at_risk(returns, 0.95),
            'downside_deviation': self._downside_deviation(returns),
            'upside_deviation': self._upside_deviation(returns),
            'skewness': stats.skew(returns.dropna()),
            'kurtosis': stats.kurtosis(returns.dropna())
        }
        metrics.update(risk_metrics)
        
        # Trading metrics
        if not transaction_history.empty:
            trading_metrics = self._analyze_trades(transaction_history)
            metrics.update(trading_metrics)
        
        # Drawdown analysis
        drawdown_metrics = self._analyze_drawdowns(portfolio_history)
        metrics.update(drawdown_metrics)
        
        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_metrics = self._compare_to_benchmark(returns, benchmark_returns)
            metrics.update(benchmark_metrics)
        
        # Time-based analysis
        time_metrics = self._analyze_time_periods(returns, portfolio_history)
        metrics.update(time_metrics)
        
        self._results = metrics
        return metrics
    
    def _calculate_returns(self, portfolio_history: pd.DataFrame) -> pd.Series:
        """Calculate period returns from portfolio history"""
        if 'total_value' not in portfolio_history.columns:
            return pd.Series()
        
        return portfolio_history['total_value'].pct_change()
    
    def _total_return(self, portfolio_history: pd.DataFrame) -> float:
        """Calculate total return"""
        if portfolio_history.empty:
            return 0.0
        
        initial_value = portfolio_history['total_value'].iloc[0]
        final_value = portfolio_history['total_value'].iloc[-1]
        
        if initial_value == 0:
            return 0.0
        
        return (final_value - initial_value) / initial_value
    
    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if returns.empty:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0.0
        
        return (1 + total_return) ** (self.periods_per_year / n_periods) - 1
    
    def _volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if returns.empty:
            return 0.0
        
        return returns.std() * np.sqrt(self.periods_per_year)
    
    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        
        volatility = returns.std()
        if volatility == 0:
            return 0.0
        
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / volatility
    
    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        downside_returns = returns[returns < 0]
        
        if downside_returns.empty:
            return 0.0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / downside_std
    
    def _max_drawdown(self, portfolio_history: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if portfolio_history.empty or 'total_value' not in portfolio_history.columns:
            return 0.0
        
        cumulative = portfolio_history['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calmar_ratio(self, returns: pd.Series, portfolio_history: pd.DataFrame) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = self._annualized_return(returns)
        max_dd = abs(self._max_drawdown(portfolio_history))
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    def _value_at_risk(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk (VaR)"""
        if returns.empty:
            return 0.0
        
        return np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    
    def _conditional_value_at_risk(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        var = self._value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < 0]
        if downside_returns.empty:
            return 0.0
        
        return downside_returns.std() * np.sqrt(self.periods_per_year)
    
    def _upside_deviation(self, returns: pd.Series) -> float:
        """Calculate upside deviation"""
        upside_returns = returns[returns > 0]
        if upside_returns.empty:
            return 0.0
        
        return upside_returns.std() * np.sqrt(self.periods_per_year)
    
    def _win_rate(self, transaction_history: pd.DataFrame) -> float:
        """Calculate win rate from transactions"""
        if transaction_history.empty:
            return 0.0
        
        # Group by instrument to find closed trades
        trades = self._extract_closed_trades(transaction_history)
        
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        return winning_trades / len(trades)
    
    def _profit_factor(self, transaction_history: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        trades = self._extract_closed_trades(transaction_history)
        
        if not trades:
            return 0.0
        
        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _average_win_loss_ratio(self, transaction_history: pd.DataFrame) -> float:
        """Calculate average win/loss ratio"""
        trades = self._extract_closed_trades(transaction_history)
        
        if not trades:
            return 0.0
        
        wins = [trade['pnl'] for trade in trades if trade['pnl'] > 0]
        losses = [abs(trade['pnl']) for trade in trades if trade['pnl'] < 0]
        
        if not wins or not losses:
            return 0.0
        
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return float('inf')
        
        return avg_win / avg_loss
    
    def _analyze_trades(self, transaction_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading statistics"""
        if transaction_history.empty:
            return {}
        
        trades = self._extract_closed_trades(transaction_history)
        
        if not trades:
            return {
                'total_trades': 0,
                'avg_trade_duration': 0,
                'avg_trades_per_day': 0
            }
        
        # Trade statistics
        pnls = [trade['pnl'] for trade in trades]
        durations = [trade['duration'].total_seconds() / 3600 for trade in trades]  # hours
        
        # Calculate trading frequency
        if 'datetime' in transaction_history.columns:
            date_range = (
                transaction_history['datetime'].max() - 
                transaction_history['datetime'].min()
            ).days
            avg_trades_per_day = len(trades) / max(date_range, 1)
        else:
            avg_trades_per_day = 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': sum(1 for pnl in pnls if pnl > 0),
            'losing_trades': sum(1 for pnl in pnls if pnl < 0),
            'avg_trade_pnl': np.mean(pnls) if pnls else 0,
            'median_trade_pnl': np.median(pnls) if pnls else 0,
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
            'avg_trade_duration': np.mean(durations) if durations else 0,
            'avg_trades_per_day': avg_trades_per_day,
            'total_commission': transaction_history['commission'].sum() if 'commission' in transaction_history.columns else 0,
            'total_slippage': transaction_history['slippage_cost'].sum() if 'slippage_cost' in transaction_history.columns else 0
        }
    
    def _analyze_drawdowns(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        if portfolio_history.empty or 'total_value' not in portfolio_history.columns:
            return {}
        
        cumulative = portfolio_history['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        # Calculate drawdown statistics
        drawdown_periods = []
        start_indices = drawdown_starts[drawdown_starts].index
        end_indices = drawdown_ends[drawdown_ends].index
        
        for i, start_idx in enumerate(start_indices):
            # Find corresponding end
            end_candidates = end_indices[end_indices > start_idx]
            if len(end_candidates) > 0:
                end_idx = end_candidates[0]
                period_drawdown = drawdown[start_idx:end_idx]
                
                drawdown_periods.append({
                    'depth': period_drawdown.min(),
                    'duration': len(period_drawdown),
                    'recovery_time': len(period_drawdown)
                })
        
        if not drawdown_periods:
            return {
                'max_drawdown_duration': 0,
                'avg_drawdown_duration': 0,
                'num_drawdowns': 0
            }
        
        return {
            'max_drawdown_duration': max(d['duration'] for d in drawdown_periods),
            'avg_drawdown_duration': np.mean([d['duration'] for d in drawdown_periods]),
            'num_drawdowns': len(drawdown_periods),
            'time_underwater': is_drawdown.sum() / len(portfolio_history) if len(portfolio_history) > 0 else 0
        }
    
    def _compare_to_benchmark(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Compare strategy performance to benchmark"""
        # Align series
        aligned_strategy, aligned_benchmark = strategy_returns.align(
            benchmark_returns, join='inner'
        )
        
        if aligned_strategy.empty:
            return {}
        
        # Calculate excess returns
        excess_returns = aligned_strategy - aligned_benchmark
        
        # Calculate beta
        covariance = np.cov(aligned_strategy.dropna(), aligned_benchmark.dropna())[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Calculate alpha (Jensen's alpha)
        strategy_annual = self._annualized_return(aligned_strategy)
        benchmark_annual = self._annualized_return(aligned_benchmark)
        alpha = strategy_annual - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))
        
        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(self.periods_per_year)
        information_ratio = (
            excess_returns.mean() * self.periods_per_year / tracking_error 
            if tracking_error != 0 else 0
        )
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': aligned_strategy.corr(aligned_benchmark),
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'excess_return': excess_returns.mean() * self.periods_per_year
        }
    
    def _analyze_time_periods(
        self,
        returns: pd.Series,
        portfolio_history: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze performance across different time periods"""
        if returns.empty:
            return {}
        
        # Monthly returns
        if hasattr(returns.index, 'to_period'):
            monthly_returns = returns.groupby(returns.index.to_period('M')).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            return {
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': (monthly_returns > 0).sum(),
                'negative_months': (monthly_returns < 0).sum(),
                'avg_monthly_return': monthly_returns.mean()
            }
        
        return {}
    
    def _extract_closed_trades(self, transaction_history: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract closed trades from transaction history"""
        if transaction_history.empty:
            return []
        
        trades = []
        
        # Group by instrument
        for instrument, group in transaction_history.groupby('instrument'):
            position = 0
            entry_price = 0
            entry_time = None
            
            for _, row in group.iterrows():
                quantity = row['quantity']
                price = row['execution_price']
                
                # Check if this closes a position
                if position != 0 and np.sign(position) != np.sign(quantity):
                    # Position is being closed or reversed
                    closed_quantity = min(abs(position), abs(quantity))
                    
                    # Calculate P&L
                    if position > 0:  # Long position
                        pnl = closed_quantity * (price - entry_price)
                    else:  # Short position
                        pnl = closed_quantity * (entry_price - price)
                    
                    # Add trade
                    trades.append({
                        'instrument': instrument,
                        'entry_time': entry_time,
                        'exit_time': row['datetime'] if 'datetime' in row else None,
                        'duration': (
                            row['datetime'] - entry_time 
                            if 'datetime' in row and entry_time else timedelta(0)
                        ),
                        'entry_price': entry_price,
                        'exit_price': price,
                        'quantity': closed_quantity,
                        'pnl': pnl
                    })
                
                # Update position
                if position == 0:
                    entry_price = price
                    entry_time = row['datetime'] if 'datetime' in row else None
                
                position += quantity
        
        return trades
    
    def get_summary_string(self) -> str:
        """Get formatted summary of performance metrics"""
        if not self._results:
            return "No results available. Run analyze() first."
        
        summary = []
        summary.append("="*60)
        summary.append("PERFORMANCE SUMMARY")
        summary.append("="*60)
        
        # Returns
        summary.append("\nRETURNS:")
        summary.append(f"  Total Return: {self._results.get('total_return', 0)*100:.2f}%")
        summary.append(f"  Annualized Return: {self._results.get('annualized_return', 0)*100:.2f}%")
        summary.append(f"  Volatility: {self._results.get('volatility', 0)*100:.2f}%")
        
        # Risk-adjusted returns
        summary.append("\nRISK-ADJUSTED RETURNS:")
        summary.append(f"  Sharpe Ratio: {self._results.get('sharpe_ratio', 0):.3f}")
        summary.append(f"  Sortino Ratio: {self._results.get('sortino_ratio', 0):.3f}")
        summary.append(f"  Calmar Ratio: {self._results.get('calmar_ratio', 0):.3f}")
        
        # Risk metrics
        summary.append("\nRISK METRICS:")
        summary.append(f"  Maximum Drawdown: {self._results.get('max_drawdown', 0)*100:.2f}%")
        summary.append(f"  Value at Risk (95%): {self._results.get('value_at_risk_95', 0)*100:.2f}%")
        summary.append(f"  Downside Deviation: {self._results.get('downside_deviation', 0)*100:.2f}%")
        
        # Trading statistics
        if 'total_trades' in self._results:
            summary.append("\nTRADING STATISTICS:")
            summary.append(f"  Total Trades: {self._results.get('total_trades', 0)}")
            summary.append(f"  Win Rate: {self._results.get('win_rate', 0)*100:.2f}%")
            summary.append(f"  Profit Factor: {self._results.get('profit_factor', 0):.2f}")
            summary.append(f"  Avg Win/Loss Ratio: {self._results.get('average_win_loss_ratio', 0):.2f}")
        
        summary.append("="*60)
        
        return "\n".join(summary)