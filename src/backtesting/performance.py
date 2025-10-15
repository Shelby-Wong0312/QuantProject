"""
Performance Analysis - Comprehensive performance metrics and risk analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies

    Calculates 30+ performance metrics including:
    - Returns and risk metrics
    - Drawdown analysis
    - Trading statistics
    - Risk-adjusted returns
    """

    def __init__(self):
        """Initialize performance analyzer"""
        self.metrics_cache = {}
        logger.debug("PerformanceAnalyzer initialized")

    def calculate_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        trades: List[Dict],
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics

        Args:
            portfolio_values: Time series of portfolio values
            returns: Time series of daily returns
            trades: List of completed trades
            initial_capital: Initial capital amount
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {}

            # Basic return metrics
            metrics.update(
                self._calculate_return_metrics(portfolio_values, returns, initial_capital)
            )

            # Risk metrics
            metrics.update(self._calculate_risk_metrics(returns))

            # Drawdown analysis
            metrics.update(self._calculate_drawdown_metrics(portfolio_values, returns))

            # Trading statistics
            metrics.update(self._calculate_trading_metrics(trades))

            # Risk-adjusted metrics
            metrics.update(self._calculate_risk_adjusted_metrics(returns))

            # Time-based metrics
            metrics.update(self._calculate_time_metrics(portfolio_values, returns))

            # Benchmark comparison (if provided)
            if benchmark_returns is not None:
                metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))

            logger.debug("Performance metrics calculated successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._get_default_metrics()

    def _calculate_return_metrics(
        self, portfolio_values: pd.Series, returns: pd.Series, initial_capital: float
    ) -> Dict[str, Any]:
        """Calculate return-based metrics"""
        metrics = {}

        if len(portfolio_values) == 0:
            return self._get_default_return_metrics()

        # Total return
        final_value = portfolio_values.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        metrics["total_return"] = total_return
        metrics["total_return_pct"] = total_return * 100

        # Annualized return
        if len(returns) > 0:
            trading_days = len(returns)
            years = trading_days / 252  # Assuming 252 trading days per year

            if years > 0:
                annual_return = (1 + total_return) ** (1 / years) - 1
                metrics["annual_return"] = annual_return
                metrics["annual_return_pct"] = annual_return * 100
            else:
                metrics["annual_return"] = 0
                metrics["annual_return_pct"] = 0
        else:
            metrics["annual_return"] = 0
            metrics["annual_return_pct"] = 0

        # Cumulative returns
        if len(returns) > 0:
            cumulative_returns = (1 + returns).cumprod() - 1
            metrics["cumulative_returns"] = cumulative_returns
        else:
            metrics["cumulative_returns"] = pd.Series()

        return metrics

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk-based metrics"""
        metrics = {}

        if len(returns) == 0:
            return self._get_default_risk_metrics()

        # Remove any infinite or NaN values
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_returns) == 0:
            return self._get_default_risk_metrics()

        # Volatility (annualized)
        daily_vol = clean_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        metrics["volatility"] = annual_vol
        metrics["volatility_pct"] = annual_vol * 100

        # Downside deviation
        negative_returns = clean_returns[clean_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            metrics["downside_deviation"] = downside_deviation
            metrics["downside_deviation_pct"] = downside_deviation * 100
        else:
            metrics["downside_deviation"] = 0
            metrics["downside_deviation_pct"] = 0

        # Value at Risk (VaR)
        metrics["var_95"] = clean_returns.quantile(0.05)
        metrics["var_99"] = clean_returns.quantile(0.01)
        metrics["var_95_pct"] = metrics["var_95"] * 100
        metrics["var_99_pct"] = metrics["var_99"] * 100

        # Conditional VaR (Expected Shortfall)
        var_95 = metrics["var_95"]
        cvar_95 = clean_returns[clean_returns <= var_95].mean()
        metrics["cvar_95"] = cvar_95
        metrics["cvar_95_pct"] = cvar_95 * 100

        # Skewness and Kurtosis
        metrics["skewness"] = clean_returns.skew()
        metrics["kurtosis"] = clean_returns.kurtosis()

        return metrics

    def _calculate_drawdown_metrics(
        self, portfolio_values: pd.Series, returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate drawdown-related metrics"""
        metrics = {}

        if len(portfolio_values) == 0:
            return {
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "avg_drawdown": 0,
                "drawdown_duration": 0,
                "recovery_time": 0,
            }

        # Calculate drawdowns
        cumulative = (1 + returns).cumprod() if len(returns) > 0 else pd.Series([1])
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdowns.min()
        metrics["max_drawdown"] = max_drawdown
        metrics["max_drawdown_pct"] = max_drawdown * 100

        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
        metrics["avg_drawdown"] = avg_drawdown
        metrics["avg_drawdown_pct"] = avg_drawdown * 100

        # Drawdown duration and recovery time
        drawdown_periods = self._identify_drawdown_periods(drawdowns)

        if drawdown_periods:
            durations = [period["duration"] for period in drawdown_periods]
            recovery_times = [
                period["recovery"] for period in drawdown_periods if period["recovery"] > 0
            ]

            metrics["max_drawdown_duration"] = max(durations) if durations else 0
            metrics["avg_drawdown_duration"] = np.mean(durations) if durations else 0
            metrics["avg_recovery_time"] = np.mean(recovery_times) if recovery_times else 0
        else:
            metrics["max_drawdown_duration"] = 0
            metrics["avg_drawdown_duration"] = 0
            metrics["avg_recovery_time"] = 0

        return metrics

    def _identify_drawdown_periods(self, drawdowns: pd.Series) -> List[Dict]:
        """Identify individual drawdown periods"""
        periods = []
        in_drawdown = False
        start_idx = 0
        peak_value = 0

        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                peak_value = drawdowns.iloc[i - 1] if i > 0 else 0
            elif dd >= 0 and in_drawdown:
                # End of drawdown (recovery)
                duration = i - start_idx
                trough_value = drawdowns.iloc[start_idx:i].min()
                recovery_time = self._calculate_recovery_time(drawdowns, i, peak_value)

                periods.append(
                    {
                        "start": start_idx,
                        "end": i,
                        "duration": duration,
                        "depth": trough_value,
                        "recovery": recovery_time,
                    }
                )

                in_drawdown = False

        # Handle case where we end in a drawdown
        if in_drawdown:
            duration = len(drawdowns) - start_idx
            trough_value = drawdowns.iloc[start_idx:].min()

            periods.append(
                {
                    "start": start_idx,
                    "end": len(drawdowns) - 1,
                    "duration": duration,
                    "depth": trough_value,
                    "recovery": 0,  # Still in drawdown
                }
            )

        return periods

    def _calculate_recovery_time(
        self, drawdowns: pd.Series, recovery_start: int, peak_value: float
    ) -> int:
        """Calculate time to recover to previous peak"""
        recovery_time = 0

        for i in range(recovery_start, len(drawdowns)):
            if drawdowns.iloc[i] >= peak_value:
                break
            recovery_time += 1

        return recovery_time

    def _calculate_trading_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate trading-related metrics"""
        metrics = {}

        if not trades:
            return self._get_default_trading_metrics()

        # Extract trade data
        pnls = [trade.get("pnl", 0) for trade in trades]

        # Basic statistics
        metrics["total_trades"] = len(trades)

        # Win/Loss analysis
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]

        metrics["winning_trades"] = len(winning_trades)
        metrics["losing_trades"] = len(losing_trades)
        metrics["win_rate"] = len(winning_trades) / len(trades) if trades else 0
        metrics["win_rate_pct"] = metrics["win_rate"] * 100

        # P&L metrics
        metrics["total_pnl"] = sum(pnls)
        metrics["avg_trade_pnl"] = np.mean(pnls) if pnls else 0
        metrics["median_trade_pnl"] = np.median(pnls) if pnls else 0

        # Win/Loss averages
        metrics["avg_win"] = np.mean(winning_trades) if winning_trades else 0
        metrics["avg_loss"] = np.mean(losing_trades) if losing_trades else 0
        metrics["largest_win"] = max(pnls) if pnls else 0
        metrics["largest_loss"] = min(pnls) if pnls else 0

        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Risk-reward ratio
        avg_win = metrics["avg_win"]
        avg_loss = abs(metrics["avg_loss"]) if metrics["avg_loss"] != 0 else 1
        metrics["risk_reward_ratio"] = avg_win / avg_loss if avg_loss > 0 else 0

        # Trade duration analysis
        durations = [trade.get("duration_hours", 0) for trade in trades]
        if durations:
            metrics["avg_trade_duration_hours"] = np.mean(durations)
            metrics["median_trade_duration_hours"] = np.median(durations)
            metrics["max_trade_duration_hours"] = max(durations)
        else:
            metrics["avg_trade_duration_hours"] = 0
            metrics["median_trade_duration_hours"] = 0
            metrics["max_trade_duration_hours"] = 0

        # Consecutive wins/losses
        metrics.update(self._calculate_consecutive_metrics(pnls))

        return metrics

    def _calculate_consecutive_metrics(self, pnls: List[float]) -> Dict[str, int]:
        """Calculate consecutive wins/losses metrics"""
        if not pnls:
            return {"max_consecutive_wins": 0, "max_consecutive_losses": 0, "current_streak": 0}

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        # Current streak
        current_streak = current_wins if current_wins > 0 else -current_losses

        return {
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
            "current_streak": current_streak,
        }

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        metrics = {}

        if len(returns) == 0:
            return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

        # Clean returns
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_returns) == 0:
            return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

        # Sharpe Ratio (assuming risk-free rate = 0)
        mean_return = clean_returns.mean()
        std_return = clean_returns.std()

        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            metrics["sharpe_ratio"] = sharpe_ratio
        else:
            metrics["sharpe_ratio"] = 0

        # Sortino Ratio
        negative_returns = clean_returns[clean_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            sortino_ratio = (mean_return / downside_std) * np.sqrt(252)
            metrics["sortino_ratio"] = sortino_ratio
        else:
            metrics["sortino_ratio"] = float("inf") if mean_return > 0 else 0

        # Calmar Ratio (requires drawdown calculation)
        cumulative = (1 + clean_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        if max_drawdown > 0:
            annual_return = mean_return * 252
            calmar_ratio = annual_return / max_drawdown
            metrics["calmar_ratio"] = calmar_ratio
        else:
            metrics["calmar_ratio"] = float("inf") if mean_return > 0 else 0

        return metrics

    def _calculate_time_metrics(
        self, portfolio_values: pd.Series, returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate time-based metrics"""
        metrics = {}

        if len(portfolio_values) == 0:
            return {"trading_days": 0, "years": 0}

        trading_days = len(portfolio_values)
        years = trading_days / 252

        metrics["trading_days"] = trading_days
        metrics["years"] = years

        # Monthly returns analysis
        if len(returns) > 30:
            try:
                # Resample to monthly
                monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

                metrics["positive_months"] = len(monthly_returns[monthly_returns > 0])
                metrics["negative_months"] = len(monthly_returns[monthly_returns < 0])
                metrics["best_month"] = monthly_returns.max()
                metrics["worst_month"] = monthly_returns.min()
                metrics["avg_monthly_return"] = monthly_returns.mean()

            except Exception:
                # If resampling fails, provide defaults
                metrics["positive_months"] = 0
                metrics["negative_months"] = 0
                metrics["best_month"] = 0
                metrics["worst_month"] = 0
                metrics["avg_monthly_return"] = 0
        else:
            metrics["positive_months"] = 0
            metrics["negative_months"] = 0
            metrics["best_month"] = 0
            metrics["worst_month"] = 0
            metrics["avg_monthly_return"] = 0

        return metrics

    def _calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics"""
        metrics = {}

        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

        if len(aligned_returns) == 0:
            return {
                "alpha": 0,
                "beta": 0,
                "correlation": 0,
                "tracking_error": 0,
                "information_ratio": 0,
            }

        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
            metrics["beta"] = beta
        else:
            beta = 0
            metrics["beta"] = 0

        # Calculate alpha
        mean_return = aligned_returns.mean() * 252
        mean_benchmark = aligned_benchmark.mean() * 252
        alpha = mean_return - beta * mean_benchmark
        metrics["alpha"] = alpha

        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        metrics["correlation"] = correlation if not np.isnan(correlation) else 0

        # Tracking error
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        metrics["tracking_error"] = tracking_error

        # Information ratio
        if tracking_error > 0:
            information_ratio = (excess_returns.mean() * 252) / tracking_error
            metrics["information_ratio"] = information_ratio
        else:
            metrics["information_ratio"] = 0

        return metrics

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics when calculation fails"""
        return {
            "total_return": 0,
            "annual_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "total_trades": 0,
            "profit_factor": 0,
        }

    def _get_default_return_metrics(self) -> Dict[str, Any]:
        """Default return metrics"""
        return {
            "total_return": 0,
            "total_return_pct": 0,
            "annual_return": 0,
            "annual_return_pct": 0,
            "cumulative_returns": pd.Series(),
        }

    def _get_default_risk_metrics(self) -> Dict[str, Any]:
        """Default risk metrics"""
        return {
            "volatility": 0,
            "volatility_pct": 0,
            "downside_deviation": 0,
            "downside_deviation_pct": 0,
            "var_95": 0,
            "var_99": 0,
            "var_95_pct": 0,
            "var_99_pct": 0,
            "cvar_95": 0,
            "cvar_95_pct": 0,
            "skewness": 0,
            "kurtosis": 0,
        }

    def _get_default_trading_metrics(self) -> Dict[str, Any]:
        """Default trading metrics"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "win_rate_pct": 0,
            "total_pnl": 0,
            "avg_trade_pnl": 0,
            "median_trade_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "profit_factor": 0,
            "risk_reward_ratio": 0,
            "avg_trade_duration_hours": 0,
            "median_trade_duration_hours": 0,
            "max_trade_duration_hours": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "current_streak": 0,
        }

    def generate_performance_report(self, metrics: Dict[str, Any]) -> str:
        """Generate formatted performance report"""
        """
PERFORMANCE ANALYSIS REPORT
===========================

RETURN METRICS
--------------
Total Return: {total_return:.2%}
Annual Return: {annual_return:.2%}
Best Month: {best_month:.2%}
Worst Month: {worst_month:.2%}

RISK METRICS
------------
Volatility: {volatility:.2%}
Sharpe Ratio: {sharpe_ratio:.2f}
Sortino Ratio: {sortino_ratio:.2f}
Max Drawdown: {max_drawdown:.2%}
VaR (95%): {var_95:.2%}

TRADING STATISTICS
------------------
Total Trades: {total_trades}
Win Rate: {win_rate:.2%}
Profit Factor: {profit_factor:.2f}
Avg Win: ${avg_win:.2f}
Avg Loss: ${avg_loss:.2f}
Risk/Reward: {risk_reward_ratio:.2f}

DRAWDOWN ANALYSIS
-----------------
Max Drawdown Duration: {max_drawdown_duration} days
Avg Recovery Time: {avg_recovery_time:.1f} days

        """.format(
            **{k: v for k, v in metrics.items()}
        )

        return report.strip()


def calculate_performance_metrics(
    portfolio_values: pd.Series, returns: pd.Series, trades: List[Dict], initial_capital: float
) -> Dict[str, Any]:
    """
    Convenience function to calculate performance metrics

    Args:
        portfolio_values: Portfolio value time series
        returns: Daily returns time series
        trades: List of trade dictionaries
        initial_capital: Initial capital amount

    Returns:
        Performance metrics dictionary
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.calculate_metrics(portfolio_values, returns, trades, initial_capital)
