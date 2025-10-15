"""
Strategy Selection System
Automated strategy selection and ensemble management
Stage 8 - Strategy Optimization Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics for selection"""

    strategy_name: str
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0

    # Advanced metrics
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    maximum_consecutive_losses: int = 0

    # Time-based performance
    monthly_returns: List[float] = field(default_factory=list)
    rolling_sharpe: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "sharpe_ratio": self.sharpe_ratio,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio,
            "volatility": self.volatility,
            "total_trades": self.total_trades,
            "alpha": self.alpha,
            "beta": self.beta,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


@dataclass
class SelectionCriteria:
    """Strategy selection criteria"""

    min_sharpe_ratio: float = 1.0
    min_annual_return: float = 0.1
    max_drawdown: float = 0.2
    min_win_rate: float = 0.4
    min_total_trades: int = 50
    max_volatility: float = 0.3
    min_profit_factor: float = 1.2

    # Advanced criteria
    min_calmar_ratio: float = 0.5
    max_var_95: float = 0.05
    max_correlation_threshold: float = 0.7

    # Time-based criteria
    min_consistency_score: float = 0.6
    lookback_periods: int = 12  # months


class StrategySelector:
    """
    Advanced strategy selection system with multiple selection methods
    """

    def __init__(
        self,
        selection_criteria: Optional[SelectionCriteria] = None,
        max_strategies: int = 5,
        rebalance_frequency: str = "monthly",
    ):
        """
        Initialize strategy selector

        Args:
            selection_criteria: Strategy selection criteria
            max_strategies: Maximum number of strategies to select
            rebalance_frequency: Frequency to rebalance strategy weights
        """
        self.criteria = selection_criteria or SelectionCriteria()
        self.max_strategies = max_strategies
        self.rebalance_frequency = rebalance_frequency

        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.selection_history: List[Dict] = []
        self.current_selection: List[str] = []
        self.strategy_weights: Dict[str, float] = {}

        logger.info(f"Strategy selector initialized with max {max_strategies} strategies")

    def add_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """
        Add strategy metrics for selection

        Args:
            metrics: Strategy performance metrics
        """
        self.strategy_metrics[metrics.strategy_name] = metrics
        logger.debug(f"Added metrics for strategy: {metrics.strategy_name}")

    def rank_strategies_by_sharpe(self) -> List[Tuple[str, float]]:
        """
        Rank strategies by Sharpe ratio

        Returns:
            List of (strategy_name, sharpe_ratio) tuples, sorted by Sharpe ratio
        """
        strategies = [
            (name, metrics.sharpe_ratio)
            for name, metrics in self.strategy_metrics.items()
            if self._meets_basic_criteria(metrics)
        ]

        return sorted(strategies, key=lambda x: x[1], reverse=True)

    def rank_strategies_by_calmar(self) -> List[Tuple[str, float]]:
        """
        Rank strategies by Calmar ratio

        Returns:
            List of (strategy_name, calmar_ratio) tuples, sorted by Calmar ratio
        """
        strategies = [
            (name, metrics.calmar_ratio)
            for name, metrics in self.strategy_metrics.items()
            if self._meets_basic_criteria(metrics)
        ]

        return sorted(strategies, key=lambda x: x[1], reverse=True)

    def rank_strategies_by_composite_score(
        self, weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank strategies by composite score

        Args:
            weights: Weights for different metrics

        Returns:
            List of (strategy_name, composite_score) tuples
        """
        if weights is None:
            weights = {
                "sharpe_ratio": 0.3,
                "annual_return": 0.2,
                "calmar_ratio": 0.2,
                "win_rate": 0.1,
                "profit_factor": 0.1,
                "consistency": 0.1,
            }

        strategies = []

        for name, metrics in self.strategy_metrics.items():
            if not self._meets_basic_criteria(metrics):
                continue

            # Normalize metrics (0-1 scale)
            normalized_metrics = self._normalize_metrics(metrics)

            # Calculate composite score
            composite_score = 0.0
            for metric, weight in weights.items():
                if metric in normalized_metrics:
                    composite_score += normalized_metrics[metric] * weight

            strategies.append((name, composite_score))

        return sorted(strategies, key=lambda x: x[1], reverse=True)

    def select_strategies_by_diversification(self, correlation_matrix: pd.DataFrame) -> List[str]:
        """
        Select strategies based on diversification (low correlation)

        Args:
            correlation_matrix: Strategy return correlation matrix

        Returns:
            List of selected strategy names
        """
        # Start with the best strategy by Sharpe ratio
        ranked_strategies = self.rank_strategies_by_sharpe()
        if not ranked_strategies:
            return []

        selected = [ranked_strategies[0][0]]

        # Add strategies with low correlation to selected ones
        for strategy_name, _ in ranked_strategies[1:]:
            if len(selected) >= self.max_strategies:
                break

            # Check correlation with already selected strategies
            max_correlation = 0.0
            for selected_strategy in selected:
                if (
                    strategy_name in correlation_matrix.index
                    and selected_strategy in correlation_matrix.columns
                ):
                    correlation = abs(correlation_matrix.loc[strategy_name, selected_strategy])
                    max_correlation = max(max_correlation, correlation)

            # Add strategy if correlation is below threshold
            if max_correlation < self.criteria.max_correlation_threshold:
                selected.append(strategy_name)

        logger.info(f"Selected {len(selected)} strategies based on diversification")
        return selected

    def select_strategies_by_machine_learning(
        self, features: pd.DataFrame, target: pd.Series
    ) -> List[str]:
        """
        Select strategies using machine learning

        Args:
            features: Strategy features for ML model
            target: Target variable (e.g., future performance)

        Returns:
            List of selected strategy names
        """
        logger.info("Selecting strategies using machine learning")

        # Prepare features
        feature_matrix = self._prepare_ml_features(features)

        # Train random forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(rf_classifier, feature_matrix, target, cv=tscv)

        logger.info(
            f"ML model cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

        # Train final model
        rf_classifier.fit(feature_matrix, target)

        # Get feature importance
        feature_importance = pd.Series(
            rf_classifier.feature_importances_, index=feature_matrix.columns
        ).sort_values(ascending=False)

        logger.info(f"Top 3 important features: {feature_importance.head(3).to_dict()}")

        # Predict strategy performance
        predictions = rf_classifier.predict_proba(feature_matrix)[
            :, 1
        ]  # Probability of good performance

        # Select top strategies based on predictions
        strategy_predictions = pd.Series(predictions, index=features.index)
        top_strategies = strategy_predictions.nlargest(self.max_strategies).index.tolist()

        return top_strategies

    def optimize_strategy_weights(
        self, returns_matrix: pd.DataFrame, method: str = "equal_weight"
    ) -> Dict[str, float]:
        """
        Optimize strategy weights for ensemble

        Args:
            returns_matrix: Strategy returns matrix
            method: Weight optimization method

        Returns:
            Dictionary of strategy weights
        """
        if method == "equal_weight":
            weights = {
                strategy: 1.0 / len(self.current_selection) for strategy in self.current_selection
            }

        elif method == "inverse_volatility":
            weights = {}
            volatilities = returns_matrix[self.current_selection].std()
            inv_vol = 1.0 / volatilities
            total_inv_vol = inv_vol.sum()

            for strategy in self.current_selection:
                weights[strategy] = inv_vol[strategy] / total_inv_vol

        elif method == "risk_parity":
            weights = self._calculate_risk_parity_weights(returns_matrix[self.current_selection])

        elif method == "mean_variance":
            weights = self._calculate_mean_variance_weights(returns_matrix[self.current_selection])

        elif method == "kelly_criterion":
            weights = self._calculate_kelly_weights(returns_matrix[self.current_selection])

        else:
            # Default to equal weight
            weights = {
                strategy: 1.0 / len(self.current_selection) for strategy in self.current_selection
            }

        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        self.strategy_weights = weights
        logger.info(f"Optimized strategy weights using {method}: {weights}")

        return weights

    def select_and_weight_strategies(
        self,
        correlation_matrix: Optional[pd.DataFrame] = None,
        returns_matrix: Optional[pd.DataFrame] = None,
        method: str = "composite_score",
        weighting_method: str = "inverse_volatility",
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select strategies and optimize weights

        Args:
            correlation_matrix: Strategy correlation matrix
            returns_matrix: Strategy returns matrix
            method: Strategy selection method
            weighting_method: Weight optimization method

        Returns:
            Tuple of (selected_strategies, weights)
        """
        logger.info(f"Selecting strategies using method: {method}")

        # Select strategies
        if method == "sharpe_ratio":
            ranked = self.rank_strategies_by_sharpe()
            selected = [name for name, _ in ranked[: self.max_strategies]]

        elif method == "calmar_ratio":
            ranked = self.rank_strategies_by_calmar()
            selected = [name for name, _ in ranked[: self.max_strategies]]

        elif method == "composite_score":
            ranked = self.rank_strategies_by_composite_score()
            selected = [name for name, _ in ranked[: self.max_strategies]]

        elif method == "diversification" and correlation_matrix is not None:
            selected = self.select_strategies_by_diversification(correlation_matrix)

        else:
            # Fallback to composite score
            ranked = self.rank_strategies_by_composite_score()
            selected = [name for name, _ in ranked[: self.max_strategies]]

        self.current_selection = selected

        # Optimize weights
        if returns_matrix is not None and selected:
            weights = self.optimize_strategy_weights(returns_matrix, weighting_method)
        else:
            weights = {strategy: 1.0 / len(selected) for strategy in selected}

        # Record selection
        self.selection_history.append(
            {
                "timestamp": datetime.now(),
                "method": method,
                "weighting_method": weighting_method,
                "selected_strategies": selected,
                "weights": weights,
                "selection_criteria": self.criteria.__dict__,
            }
        )

        logger.info(f"Selected {len(selected)} strategies: {selected}")

        return selected, weights

    def _meets_basic_criteria(self, metrics: StrategyMetrics) -> bool:
        """Check if strategy meets basic selection criteria"""
        return (
            metrics.sharpe_ratio >= self.criteria.min_sharpe_ratio
            and metrics.annual_return >= self.criteria.min_annual_return
            and metrics.max_drawdown <= self.criteria.max_drawdown
            and metrics.win_rate >= self.criteria.min_win_rate
            and metrics.total_trades >= self.criteria.min_total_trades
            and metrics.volatility <= self.criteria.max_volatility
            and metrics.profit_factor >= self.criteria.min_profit_factor
        )

    def _normalize_metrics(self, metrics: StrategyMetrics) -> Dict[str, float]:
        """Normalize metrics to 0-1 scale"""
        all_metrics = list(self.strategy_metrics.values())

        # Get min/max for normalization
        sharpe_values = [m.sharpe_ratio for m in all_metrics]
        return_values = [m.annual_return for m in all_metrics]
        calmar_values = [m.calmar_ratio for m in all_metrics]
        win_rate_values = [m.win_rate for m in all_metrics]
        profit_factor_values = [m.profit_factor for m in all_metrics]

        normalized = {}

        if sharpe_values:
            min_sharpe, max_sharpe = min(sharpe_values), max(sharpe_values)
            if max_sharpe > min_sharpe:
                normalized["sharpe_ratio"] = (metrics.sharpe_ratio - min_sharpe) / (
                    max_sharpe - min_sharpe
                )
            else:
                normalized["sharpe_ratio"] = 1.0

        if return_values:
            min_return, max_return = min(return_values), max(return_values)
            if max_return > min_return:
                normalized["annual_return"] = (metrics.annual_return - min_return) / (
                    max_return - min_return
                )
            else:
                normalized["annual_return"] = 1.0

        if calmar_values:
            min_calmar, max_calmar = min(calmar_values), max(calmar_values)
            if max_calmar > min_calmar:
                normalized["calmar_ratio"] = (metrics.calmar_ratio - min_calmar) / (
                    max_calmar - min_calmar
                )
            else:
                normalized["calmar_ratio"] = 1.0

        normalized["win_rate"] = metrics.win_rate  # Already 0-1

        if profit_factor_values:
            min_pf, max_pf = min(profit_factor_values), max(profit_factor_values)
            if max_pf > min_pf:
                normalized["profit_factor"] = (metrics.profit_factor - min_pf) / (max_pf - min_pf)
            else:
                normalized["profit_factor"] = 1.0

        # Consistency score (simplified)
        normalized["consistency"] = min(metrics.win_rate * metrics.profit_factor / 2, 1.0)

        return normalized

    def _prepare_ml_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        # Add derived features
        feature_matrix = features.copy()

        # Risk-adjusted returns
        if "annual_return" in features.columns and "volatility" in features.columns:
            feature_matrix["risk_adjusted_return"] = (
                features["annual_return"] / features["volatility"]
            )

        # Profit per trade
        if "total_return" in features.columns and "total_trades" in features.columns:
            feature_matrix["profit_per_trade"] = features["total_return"] / features["total_trades"]

        # Consistency metrics
        if "win_rate" in features.columns and "profit_factor" in features.columns:
            feature_matrix["consistency_score"] = features["win_rate"] * features["profit_factor"]

        return feature_matrix.fillna(0)

    def _calculate_risk_parity_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk parity weights"""
        # Simplified risk parity: inverse volatility with risk budget adjustment
        volatilities = returns.std()
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()

        return weights.to_dict()

    def _calculate_mean_variance_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean-variance optimal weights"""
        # Simplified mean-variance optimization
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Add small regularization to diagonal
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8

        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((len(mean_returns), 1))

            # Calculate optimal weights
            weights = inv_cov @ mean_returns.values
            weights = weights / np.sum(weights)

            # Ensure non-negative weights
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)

            return dict(zip(returns.columns, weights))

        except np.linalg.LinAlgError:
            # Fallback to equal weights
            n_strategies = len(returns.columns)
            return {col: 1.0 / n_strategies for col in returns.columns}

    def _calculate_kelly_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate Kelly criterion weights"""
        weights = {}

        for strategy in returns.columns:
            strategy_returns = returns[strategy].dropna()

            if len(strategy_returns) == 0:
                weights[strategy] = 0.0
                continue

            # Calculate Kelly fraction
            mean_return = strategy_returns.mean()
            variance = strategy_returns.var()

            if variance > 0:
                kelly_fraction = mean_return / variance
                # Cap Kelly fraction to prevent extreme positions
                kelly_fraction = np.clip(kelly_fraction, 0, 0.25)
            else:
                kelly_fraction = 0.0

            weights[strategy] = kelly_fraction

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            n_strategies = len(weights)
            weights = {k: 1.0 / n_strategies for k in weights.keys()}

        return weights

    def evaluate_selection_performance(
        self, actual_returns: pd.DataFrame, benchmark_return: float = 0.0
    ) -> Dict[str, float]:
        """
        Evaluate performance of strategy selection

        Args:
            actual_returns: Actual returns of selected strategies
            benchmark_return: Benchmark return for comparison

        Returns:
            Performance evaluation metrics
        """
        if not self.current_selection or not self.strategy_weights:
            return {}

        # Calculate ensemble return
        ensemble_returns = []
        for date in actual_returns.index:
            ensemble_return = 0.0
            for strategy in self.current_selection:
                if strategy in actual_returns.columns:
                    weight = self.strategy_weights.get(strategy, 0)
                    strategy_return = actual_returns.loc[date, strategy]
                    ensemble_return += weight * strategy_return
            ensemble_returns.append(ensemble_return)

        ensemble_series = pd.Series(ensemble_returns, index=actual_returns.index)

        # Calculate performance metrics
        total_return = ensemble_series.sum()
        volatility = ensemble_series.std()
        sharpe_ratio = ensemble_series.mean() / volatility if volatility > 0 else 0

        # Drawdown calculation
        cumulative_returns = (1 + ensemble_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        return {
            "ensemble_total_return": total_return,
            "ensemble_volatility": volatility,
            "ensemble_sharpe_ratio": sharpe_ratio,
            "ensemble_max_drawdown": max_drawdown,
            "excess_return_vs_benchmark": total_return - benchmark_return,
            "number_of_strategies": len(self.current_selection),
            "weight_concentration": (
                max(self.strategy_weights.values()) if self.strategy_weights else 0
            ),
        }

    def save_selection_results(self, filepath: str = "reports/strategy_selection_results.json"):
        """
        Save strategy selection results

        Args:
            filepath: Path to save results
        """
        results = {
            "selection_criteria": self.criteria.__dict__,
            "current_selection": self.current_selection,
            "strategy_weights": self.strategy_weights,
            "strategy_metrics": {
                name: metrics.to_dict() for name, metrics in self.strategy_metrics.items()
            },
            "selection_history": self.selection_history[-10:],  # Last 10 selections
            "timestamp": datetime.now().isoformat(),
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Strategy selection results saved to {filepath}")

    def get_selection_summary(self) -> Dict[str, Any]:
        """Get strategy selection summary"""
        return {
            "total_strategies_evaluated": len(self.strategy_metrics),
            "selected_strategies": self.current_selection,
            "strategy_weights": self.strategy_weights,
            "selection_criteria": self.criteria.__dict__,
            "last_selection_date": (
                self.selection_history[-1]["timestamp"] if self.selection_history else None
            ),
        }


def create_sample_strategy_metrics() -> List[StrategyMetrics]:
    """Create sample strategy metrics for testing"""
    strategies = []

    for i in range(10):
        metrics = StrategyMetrics(
            strategy_name=f"Strategy_{i+1}",
            sharpe_ratio=np.random.uniform(0.5, 2.5),
            annual_return=np.random.uniform(0.05, 0.3),
            max_drawdown=np.random.uniform(0.1, 0.4),
            win_rate=np.random.uniform(0.3, 0.7),
            profit_factor=np.random.uniform(0.8, 2.0),
            calmar_ratio=np.random.uniform(0.2, 1.5),
            volatility=np.random.uniform(0.1, 0.4),
            total_trades=np.random.randint(50, 500),
        )
        strategies.append(metrics)

    return strategies


async def main():
    """Test strategy selector"""
    print("\n" + "=" * 70)
    print("STRATEGY SELECTION SYSTEM")
    print("Stage 8 - Strategy Optimization Framework")
    print("=" * 70)

    # Create sample data
    sample_metrics = create_sample_strategy_metrics()

    # Initialize selector
    criteria = SelectionCriteria(
        min_sharpe_ratio=1.0, min_annual_return=0.1, max_drawdown=0.3, min_win_rate=0.4
    )

    selector = StrategySelector(selection_criteria=criteria, max_strategies=3)

    # Add strategy metrics
    for metrics in sample_metrics:
        selector.add_strategy_metrics(metrics)

    print(f"\nEvaluated {len(sample_metrics)} strategies")

    # Test different selection methods
    print("\n1. Ranking by Sharpe Ratio:")
    sharpe_ranking = selector.rank_strategies_by_sharpe()
    for i, (name, score) in enumerate(sharpe_ranking[:5], 1):
        print(f"  {i}. {name}: {score:.3f}")

    print("\n2. Ranking by Composite Score:")
    composite_ranking = selector.rank_strategies_by_composite_score()
    for i, (name, score) in enumerate(composite_ranking[:5], 1):
        print(f"  {i}. {name}: {score:.3f}")

    # Select and weight strategies
    selected, weights = selector.select_and_weight_strategies(
        method="composite_score", weighting_method="inverse_volatility"
    )

    print(f"\n3. Final Selection ({len(selected)} strategies):")
    for strategy in selected:
        weight = weights.get(strategy, 0)
        print(f"  {strategy}: {weight:.1%}")

    # Save results
    selector.save_selection_results()

    # Print summary
    summary = selector.get_selection_summary()
    print("\nSelection Summary:")
    print(f"Total strategies evaluated: {summary['total_strategies_evaluated']}")
    print(f"Selected strategies: {len(summary['selected_strategies'])}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
