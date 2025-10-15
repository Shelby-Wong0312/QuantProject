"""
Portfolio Data Adapter
連接投資組合環境與視覺化儀表板
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from rl_trading.environments.portfolio_env import PortfolioTradingEnvironment
from sensory_models.relation_analyzer import StockRelationAnalyzer

logger = logging.getLogger(__name__)


class PortfolioDataAdapter:
    """
    Adapter to connect portfolio environment data to visualization dashboard
    """

    def __init__(
        self,
        portfolio_env: Optional[PortfolioTradingEnvironment] = None,
        relation_analyzer: Optional[StockRelationAnalyzer] = None,
    ):
        """
        Initialize data adapter

        Args:
            portfolio_env: Portfolio trading environment
            relation_analyzer: Stock relation analyzer for GNN features
        """
        self.portfolio_env = portfolio_env
        self.relation_analyzer = relation_analyzer
        self.cache = {}

    def get_portfolio_data(
        self, time_range: str = "3M", include_gnn: bool = True
    ) -> Dict[str, Any]:
        """
        Get portfolio data for visualization

        Args:
            time_range: Time range to fetch ('1W', '1M', '3M', '6M', '1Y', 'ALL')
            include_gnn: Whether to include GNN analysis

        Returns:
            Dictionary containing all portfolio data
        """
        if not self.portfolio_env:
            return self._generate_sample_data(time_range)

        try:
            # Get environment data
            env_data = self._extract_env_data(time_range)

            # Add GNN analysis if requested
            if include_gnn and self.relation_analyzer:
                env_data["gnn_analysis"] = self._get_gnn_analysis()

            return env_data

        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return self._generate_sample_data(time_range)

    def _extract_env_data(self, time_range: str) -> Dict[str, Any]:
        """Extract data from portfolio environment"""
        # Get portfolio history
        history = self.portfolio_env.portfolio_history

        if not history:
            return self._generate_sample_data(time_range)

        # Convert history to DataFrame
        history_df = pd.DataFrame(history)

        # Filter by time range
        history_df = self._filter_by_time_range(history_df, time_range)

        # Extract data
        dates = history_df["timestamp"].tolist()
        portfolio_values = history_df["total_value"].tolist()

        # Calculate returns
        portfolio_returns = history_df["total_value"].pct_change().fillna(0).tolist()

        # Get symbols
        symbols = self.portfolio_env.symbols

        # Extract individual asset values if available
        asset_values = {}
        for symbol in symbols:
            if f"{symbol}_value" in history_df.columns:
                asset_values[symbol] = history_df[f"{symbol}_value"].tolist()

        # Extract positions over time
        positions_data = []
        for _, row in history_df.iterrows():
            position = {}
            for symbol in symbols:
                if f"{symbol}_weight" in row:
                    position[symbol] = row[f"{symbol}_weight"]
                else:
                    position[symbol] = 0
            position["Cash"] = row.get("cash_weight", 0)
            positions_data.append(position)

        # Get trading records
        trades = self._extract_trading_records()

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(symbols)

        return {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "portfolio_returns": portfolio_returns,
            "asset_values": asset_values,
            "benchmark_values": self._get_benchmark_values(len(dates)),
            "positions": positions_data,
            "correlation_matrix": correlation_matrix.tolist(),
            "symbols": symbols,
            "trades": trades,
        }

    def _filter_by_time_range(self, df: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """Filter dataframe by time range"""
        if time_range == "ALL":
            return df

        period_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}

        days = period_map.get(time_range, 90)
        cutoff_date = datetime.now() - timedelta(days=days)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df[df["timestamp"] >= cutoff_date]

        return df

    def _extract_trading_records(self) -> List[Dict[str, Any]]:
        """Extract trading records from environment"""
        if not hasattr(self.portfolio_env, "trade_history"):
            return []

        trades = []
        for trade in self.portfolio_env.trade_history[-50:]:  # Last 50 trades
            trades.append(
                {
                    "date": trade["timestamp"].strftime("%Y-%m-%d"),
                    "symbol": trade["symbol"],
                    "action": trade["action"],
                    "quantity": trade["quantity"],
                    "price": trade["price"],
                    "commission": trade.get("commission", 0),
                }
            )

        return trades

    def _calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Calculate correlation matrix from price data"""
        if not self.portfolio_env.market_data:
            # Return random correlation matrix
            n = len(symbols)
            corr = np.random.rand(n, n)
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)
            return corr

        # Calculate returns for each symbol
        returns_data = {}
        for symbol in symbols:
            if symbol in self.portfolio_env.market_data:
                prices = self.portfolio_env.market_data[symbol]["close"]
                returns = prices.pct_change().dropna()
                returns_data[symbol] = returns

        # Create correlation matrix
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            return returns_df.corr().values
        else:
            n = len(symbols)
            return np.eye(n)

    def _get_benchmark_values(self, length: int) -> List[float]:
        """Get benchmark values"""
        # In production, fetch real benchmark data
        # For now, generate synthetic benchmark
        returns = np.random.normal(0.0007, 0.012, length)
        values = 100000 * np.exp(np.cumsum(returns))
        return values.tolist()

    def _get_gnn_analysis(self) -> Dict[str, Any]:
        """Get GNN analysis results"""
        if not self.relation_analyzer:
            return {}

        try:
            # Get latest analysis from cache
            if "latest" in self.relation_analyzer.analysis_cache:
                analysis = self.relation_analyzer.analysis_cache["latest"]

                return {
                    "clusters": analysis.get("clusters", {}),
                    "key_relations": analysis.get("key_relations", []),
                    "centrality": analysis.get("centrality", {}),
                    "graph_stats": analysis.get("graph_stats", {}),
                }
        except Exception as e:
            logger.error(f"Error getting GNN analysis: {e}")

        return {}

    def _generate_sample_data(self, time_range: str) -> Dict[str, Any]:
        """Generate sample data for testing"""
        # Determine number of data points
        period_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 252, "ALL": 500}
        n_days = period_map.get(time_range, 90)

        # Generate dates
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

        # Sample symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        n_assets = len(symbols)

        # Generate portfolio returns
        portfolio_returns = np.random.normal(0.001, 0.015, n_days)
        portfolio_values = 100000 * np.exp(np.cumsum(portfolio_returns))

        # Generate asset values
        asset_values = {}
        for symbol in symbols:
            returns = np.random.normal(0.0008, 0.02, n_days)
            values = 100000 / n_assets * np.exp(np.cumsum(returns))
            asset_values[symbol] = values.tolist()

        # Generate benchmark
        benchmark_returns = np.random.normal(0.0007, 0.012, n_days)
        benchmark_values = 100000 * np.exp(np.cumsum(benchmark_returns))

        # Generate positions
        positions = []
        for i in range(n_days):
            weights = np.random.dirichlet(np.ones(n_assets + 1))
            position = {symbols[j]: weights[j] for j in range(n_assets)}
            position["Cash"] = weights[-1]
            positions.append(position)

        # Generate correlation matrix
        corr_matrix = np.random.rand(n_assets, n_assets)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)

        # Generate sample trades
        trades = []
        trade_dates = np.random.choice(dates, size=min(20, n_days), replace=False)
        for date in sorted(trade_dates):
            trades.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": np.random.choice(symbols),
                    "action": np.random.choice(["Buy", "Sell"]),
                    "quantity": np.random.randint(10, 100),
                    "price": np.round(np.random.uniform(50, 200), 2),
                    "commission": np.round(np.random.uniform(1, 10), 2),
                }
            )

        return {
            "dates": dates.strftime("%Y-%m-%d").tolist(),
            "portfolio_values": portfolio_values.tolist(),
            "portfolio_returns": portfolio_returns.tolist(),
            "asset_values": asset_values,
            "benchmark_values": benchmark_values.tolist(),
            "positions": positions,
            "correlation_matrix": corr_matrix.tolist(),
            "symbols": symbols,
            "trades": trades,
        }

    def calculate_risk_metrics(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary of risk metrics
        """
        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return {
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "volatility": 0,
                "var_95": 0,
                "cvar_95": 0,
                "beta": 0,
                "calmar_ratio": 0,
            }

        # Basic metrics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino_ratio = (mean_return - risk_free_rate / 252) / downside_std * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Volatility (annualized)
        volatility = std_return * np.sqrt(252)

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])

        # Calmar ratio
        calmar_ratio = mean_return * 252 / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "calmar_ratio": calmar_ratio,
        }


# Singleton instance for global access
_adapter_instance = None


def get_portfolio_adapter() -> PortfolioDataAdapter:
    """Get global portfolio data adapter instance"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = PortfolioDataAdapter()
    return _adapter_instance


def set_portfolio_environment(
    env: PortfolioTradingEnvironment, analyzer: Optional[StockRelationAnalyzer] = None
):
    """Set portfolio environment for the adapter"""
    adapter = get_portfolio_adapter()
    adapter.portfolio_env = env
    adapter.relation_analyzer = analyzer
