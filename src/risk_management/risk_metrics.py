"""
Risk Metrics Calculation Module
VaR, CVaR, Maximum Drawdown
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


class RiskMetrics:
    """Risk metrics calculator for portfolio analysis"""

    @staticmethod
    def calculate_var(
        returns: Union[list, np.ndarray, pd.Series], confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: Return series
            confidence: Confidence level (default 0.95)

        Returns:
            VaR value
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        elif isinstance(returns, list):
            returns = np.array(returns)

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_cvar(
        returns: Union[list, np.ndarray, pd.Series], confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)

        Args:
            returns: Return series
            confidence: Confidence level (default 0.95)

        Returns:
            CVaR value
        """
        var = RiskMetrics.calculate_var(returns, confidence)

        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        elif isinstance(returns, list):
            returns = np.array(returns)

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # Returns worse than VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        return np.mean(tail_returns)

    @staticmethod
    def calculate_max_drawdown(
        prices: Union[list, np.ndarray, pd.Series],
    ) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown

        Args:
            prices: Price series

        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if isinstance(prices, (list, np.ndarray)):
            prices = pd.Series(prices)

        prices = prices.dropna()

        if len(prices) == 0:
            return 0.0, 0, 0

        # Calculate cumulative maximum
        cummax = prices.cummax()

        # Calculate drawdown
        drawdown = (prices - cummax) / cummax

        # Find maximum drawdown
        max_dd = drawdown.min()

        # Find start and end indices
        end_idx = drawdown.idxmin()
        start_idx = prices[:end_idx].idxmax()

        return abs(max_dd), start_idx, end_idx

    @staticmethod
    def calculate_sharpe_ratio(
        returns: Union[list, np.ndarray, pd.Series], risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Sharpe ratio
        """
        if isinstance(returns, (list, np.ndarray)):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_sortino_ratio(
        returns: Union[list, np.ndarray, pd.Series], risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio

        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Sortino ratio
        """
        if isinstance(returns, (list, np.ndarray)):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf")

        downside_std = downside_returns.std()

        if downside_std == 0:
            return float("inf")

        return excess_returns.mean() / downside_std * np.sqrt(252)
