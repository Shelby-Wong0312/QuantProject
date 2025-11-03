"""
Hyperparameter Tuning System
Bayesian optimization for ML/DL/RL model parameters
Cloud Quant - Task Q-701
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import asyncio
from datetime import datetime
import itertools
import warnings

warnings.filterwarnings("ignore")


# Simplified Bayesian optimization (without external dependencies)
class SimpleBayesianOptimizer:
    """Simplified Bayesian Optimizer"""

    def __init__(self, param_bounds: Dict, n_iter: int = 50):
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.history = []

    def optimize(self, objective_func):
        """Run optimization"""
        None
        best_score = -np.inf

        for i in range(self.n_iter):
            # Sample parameters
            params = {}
            for param, bounds in self.param_bounds.items():
                if isinstance(bounds, tuple):
                    params[param] = np.random.uniform(bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    params[param] = np.random.choice(bounds)

            # Evaluate
            score = objective_func(params)
            self.history.append({"params": params, "score": score})

            if score > best_score:
                best_score = score
                params

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{self.n_iter}: Best Score = {best_score:.4f}")

        return best_params, best_score


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration"""

    target_metric: str = "sharpe_ratio"  # sharpe_ratio, annual_return, calmar_ratio
    n_iterations: int = 50
    n_random_starts: int = 10
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    use_parallel: bool = True
    save_results: bool = True


@dataclass
class ParameterSpace:
    """Parameter search space definition"""

    # LSTM parameters
    lstm_hidden_dim: Tuple[int, int] = (64, 256)
    lstm_num_layers: Tuple[int, int] = (2, 5)
    lstm_dropout: Tuple[float, float] = (0.1, 0.5)
    lstm_learning_rate: Tuple[float, float] = (1e-5, 1e-2)

    # XGBoost parameters
    xgb_n_estimators: Tuple[int, int] = (50, 500)
    xgb_max_depth: Tuple[int, int] = (3, 10)
    xgb_learning_rate: Tuple[float, float] = (0.01, 0.3)
    xgb_subsample: Tuple[float, float] = (0.6, 1.0)
    xgb_colsample_bytree: Tuple[float, float] = (0.6, 1.0)

    # PPO parameters
    ppo_learning_rate: Tuple[float, float] = (1e-5, 1e-3)
    ppo_gamma: Tuple[float, float] = (0.9, 0.999)
    ppo_clip_ratio: Tuple[float, float] = (0.1, 0.3)
    ppo_batch_size: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # Strategy parameters
    model_weights_lstm: Tuple[float, float] = (0.2, 0.5)
    model_weights_xgb: Tuple[float, float] = (0.2, 0.5)
    model_weights_ppo: Tuple[float, float] = (0.1, 0.4)

    # Risk parameters
    risk_tolerance: Tuple[float, float] = (0.01, 0.05)
    max_positions: Tuple[int, int] = (5, 30)
    stop_loss_multiplier: Tuple[float, float] = (1.5, 3.0)


class HyperparameterTuner:
    """
    Hyperparameter tuning system for ML/DL/RL trading strategies
    Uses Bayesian optimization to find optimal parameters
    """

    def __init__(
        self, config: OptimizationConfig, param_space: Optional[ParameterSpace] = None
    ):
        """
        Initialize hyperparameter tuner

        Args:
            config: Optimization configuration
            param_space: Parameter search space
        """
        self.config = config
        self.param_space = param_space or ParameterSpace()

        # Results storage
        self.optimization_history: List[Dict] = []
        self.best_params: Dict = {}
        self.best_score: float = -np.inf

        logger.info(
            f"Hyperparameter Tuner initialized with target metric: {config.target_metric}"
        )

    def create_search_space(self) -> Dict:
        """
        Create parameter search space for optimization

        Returns:
            Dictionary of parameter bounds
        """
        search_space = {
            # LSTM parameters
            "lstm_hidden_dim": self.param_space.lstm_hidden_dim,
            "lstm_num_layers": self.param_space.lstm_num_layers,
            "lstm_dropout": self.param_space.lstm_dropout,
            "lstm_learning_rate": self.param_space.lstm_learning_rate,
            # XGBoost parameters
            "xgb_n_estimators": self.param_space.xgb_n_estimators,
            "xgb_max_depth": self.param_space.xgb_max_depth,
            "xgb_learning_rate": self.param_space.xgb_learning_rate,
            "xgb_subsample": self.param_space.xgb_subsample,
            # PPO parameters
            "ppo_learning_rate": self.param_space.ppo_learning_rate,
            "ppo_gamma": self.param_space.ppo_gamma,
            "ppo_clip_ratio": self.param_space.ppo_clip_ratio,
            # Strategy parameters
            "model_weight_lstm": self.param_space.model_weights_lstm,
            "model_weight_xgb": self.param_space.model_weights_xgb,
            "model_weight_ppo": self.param_space.model_weights_ppo,
            # Risk parameters
            "risk_tolerance": self.param_space.risk_tolerance,
            "stop_loss_multiplier": self.param_space.stop_loss_multiplier,
        }

        return search_space

    async def optimize_ml_parameters(
        self,
        historical_data: Dict[str, pd.DataFrame],
        validation_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict:
        """
        Optimize ML model parameters using Bayesian optimization

        Args:
            historical_data: Training data
            validation_data: Validation data (optional)

        Returns:
            Dictionary of optimal parameters
        """
        logger.info("Starting hyperparameter optimization...")

        # Create search space
        search_space = self.create_search_space()

        # Define objective function
        async def objective(params: Dict) -> float:
            """Objective function to maximize"""
            try:
                # Create strategy with given parameters
                from src.strategies.ml_strategy_integration import MLStrategyIntegration
                from src.backtesting.ml_backtest import MLBacktester, BacktestConfig

                # Initialize strategy with parameters
                strategy = MLStrategyIntegration(
                    initial_capital=100000,
                    risk_tolerance=params.get("risk_tolerance", 0.02),
                    max_positions=20,
                )

                # Update model weights
                total_weight = (
                    params.get("model_weight_lstm", 0.33)
                    + params.get("model_weight_xgb", 0.33)
                    + params.get("model_weight_ppo", 0.34)
                )

                strategy.model_weights = {
                    "lstm": params.get("model_weight_lstm", 0.33) / total_weight,
                    "xgboost": params.get("model_weight_xgb", 0.33) / total_weight,
                    "ppo": params.get("model_weight_ppo", 0.34) / total_weight,
                }

                # Configure stop loss
                strategy.stop_loss.atr_multiplier = params.get(
                    "stop_loss_multiplier", 2.0
                )

                # Run simplified backtest
                backtest_config = BacktestConfig(
                    initial_capital=100000,
                    rebalance_frequency="monthly",
                    use_walk_forward=False,  # Faster for optimization
                )

                backtester = MLBacktester(backtest_config)
                backtester.historical_data = historical_data

                # Run backtest
                result = await backtester.backtest_strategy(historical_data, strategy)

                # Return target metric
                if self.config.target_metric == "sharpe_ratio":
                    return result.sharpe_ratio
                elif self.config.target_metric == "annual_return":
                    return result.annual_return
                elif self.config.target_metric == "calmar_ratio":
                    return result.calmar_ratio
                else:
                    return result.sharpe_ratio

            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return -np.inf

        # Wrapper for sync objective
        def sync_objective(params: Dict) -> float:
            """Synchronous wrapper for async objective"""
            return asyncio.run(objective(params))

        # Run Bayesian optimization
        optimizer = SimpleBayesianOptimizer(
            param_bounds=search_space, n_iter=self.config.n_iterations
        )

        print(
            f"\nOptimizing {self.config.target_metric} with {self.config.n_iterations} iterations..."
        )
        best_params, best_score = optimizer.optimize(sync_objective)

        # Store results
        self.best_params = best_params
        self.best_score = best_score
        self.optimization_history = optimizer.history

        # Save results
        if self.config.save_results:
            self.save_optimization_results()

        logger.info(
            f"Optimization complete. Best {self.config.target_metric}: {best_score:.4f}"
        )

        return best_params

    def grid_search(
        self, param_grid: Dict, historical_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Perform grid search for comparison

        Args:
            param_grid: Parameter grid
            historical_data: Historical data

        Returns:
            Best parameters from grid search
        """
        logger.info("Running grid search...")

        # Create parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        None
        best_score = -np.inf

        # Iterate through all combinations
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))

            # Evaluate parameters (simplified)
            score = self._evaluate_params_simple(params, historical_data)

            if score > best_score:
                best_score = score
                params

            self.optimization_history.append(
                {"params": params, "score": score, "method": "grid_search"}
            )

        logger.info(f"Grid search complete. Best score: {best_score:.4f}")

        return best_params

    def random_search(
        self, n_iter: int, historical_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Perform random search

        Args:
            n_iter: Number of iterations
            historical_data: Historical data

        Returns:
            Best parameters from random search
        """
        logger.info(f"Running random search with {n_iter} iterations...")

        search_space = self.create_search_space()

        None
        best_score = -np.inf

        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for param, bounds in search_space.items():
                if isinstance(bounds, tuple):
                    if param.endswith("learning_rate"):
                        # Log-uniform for learning rates
                        params[param] = 10 ** np.random.uniform(
                            np.log10(bounds[0]), np.log10(bounds[1])
                        )
                    elif any(
                        param.endswith(x)
                        for x in ["dim", "layers", "estimators", "depth", "positions"]
                    ):
                        # Integer parameters
                        params[param] = np.random.randint(bounds[0], bounds[1] + 1)
                    else:
                        # Uniform for other parameters
                        params[param] = np.random.uniform(bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    params[param] = np.random.choice(bounds)

            # Evaluate
            score = self._evaluate_params_simple(params, historical_data)

            if score > best_score:
                best_score = score
                params

            self.optimization_history.append(
                {"params": params, "score": score, "method": "random_search"}
            )

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Random search iteration {i+1}/{n_iter}: Best score = {best_score:.4f}"
                )

        return best_params

    def _evaluate_params_simple(
        self, params: Dict, historical_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Simple parameter evaluation for quick testing

        Args:
            params: Parameters to evaluate
            historical_data: Historical data

        Returns:
            Score (simplified calculation)
        """
        # Simplified scoring based on parameter values
        # In practice, this would run a full backtest

        score = 0.0

        # Favor balanced model weights
        model_weights = [
            params.get("model_weight_lstm", 0.33),
            params.get("model_weight_xgb", 0.33),
            params.get("model_weight_ppo", 0.34),
        ]
        weight_std = np.std(model_weights)
        score += (0.5 - weight_std) * 2  # Penalty for imbalanced weights

        # Favor moderate risk tolerance
        risk_tol = params.get("risk_tolerance", 0.02)
        score += -abs(risk_tol - 0.025) * 10  # Optimal around 2.5%

        # Favor reasonable stop loss
        stop_loss = params.get("stop_loss_multiplier", 2.0)
        score += -abs(stop_loss - 2.0) * 0.5  # Optimal around 2.0

        # Add some randomness (simulating backtest variance)
        score += np.random.normal(0, 0.1)

        return score

    def analyze_optimization_results(self) -> Dict:
        """
        Analyze optimization results and provide insights

        Returns:
            Analysis dictionary
        """
        if not self.optimization_history:
            return {}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.optimization_history)

        # Extract scores
        scores = df["score"].values

        # Parameter importance (simplified - correlation with score)
        param_importance = {}

        for param in self.best_params.keys():
            if param in df["params"].iloc[0]:
                param_values = [p.get(param, 0) for p in df["params"]]
                if isinstance(param_values[0], (int, float)):
                    correlation = np.corrcoef(param_values, scores)[0, 1]
                    param_importance[param] = abs(correlation)

        # Sort by importance
        param_importance = dict(
            sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        )

        analysis = {
            "total_evaluations": len(self.optimization_history),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
            },
            "parameter_importance": param_importance,
            "convergence": self._analyze_convergence(),
            "top_5_configurations": self._get_top_configurations(5),
        }

        return analysis

    def _analyze_convergence(self) -> Dict:
        """Analyze optimization convergence"""
        if len(self.optimization_history) < 10:
            return {}

        scores = [h["score"] for h in self.optimization_history]

        # Calculate running best
        running_best = []
        current_best = -np.inf
        for score in scores:
            current_best = max(current_best, score)
            running_best.append(current_best)

        # Check if converged (best score hasn't improved in last 20% of iterations)
        n_check = max(1, len(scores) // 5)
        converged = all(running_best[-n_check:] == running_best[-1])

        return {
            "converged": converged,
            "iterations_to_best": running_best.index(self.best_score) + 1,
            "final_improvement_rate": (running_best[-1] - running_best[0])
            / len(running_best),
        }

    def _get_top_configurations(self, n: int) -> List[Dict]:
        """Get top N configurations"""
        sorted_history = sorted(
            self.optimization_history, key=lambda x: x["score"], reverse=True
        )
        return sorted_history[:n]

    def save_optimization_results(
        self, filepath: str = "reports/optimal_parameters.yaml"
    ):
        """
        Save optimization results to file

        Args:
            filepath: Path to save results
        """
        results = {
            "optimization_config": {
                "target_metric": self.config.target_metric,
                "n_iterations": self.config.n_iterations,
                "timestamp": datetime.now().isoformat(),
            },
            "best_parameters": self.best_params,
            "best_score": float(self.best_score),
            "analysis": self.analyze_optimization_results(),
            "optimization_history": self.optimization_history[:10],  # Save top 10 only
        }

        # Save as JSON (more universal than YAML)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath.replace(".yaml", ".json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Optimization results saved to {filepath}")

        # Also save simplified version as YAML-like format
        with open(filepath, "w") as f:
            f.write("# Optimal Parameters for ML/DL/RL Trading Strategy\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Target Metric: {self.config.target_metric}\n")
            f.write(f"# Best Score: {self.best_score:.4f}\n\n")

            f.write("# LSTM Parameters\n")
            for key, value in self.best_params.items():
                if key.startswith("lstm_"):
                    f.write(f"{key}: {value}\n")

            f.write("\n# XGBoost Parameters\n")
            for key, value in self.best_params.items():
                if key.startswith("xgb_"):
                    f.write(f"{key}: {value}\n")

            f.write("\n# PPO Parameters\n")
            for key, value in self.best_params.items():
                if key.startswith("ppo_"):
                    f.write(f"{key}: {value}\n")

            f.write("\n# Strategy Parameters\n")
            for key, value in self.best_params.items():
                if key.startswith("model_weight"):
                    f.write(f"{key}: {value}\n")

            f.write("\n# Risk Parameters\n")
            for key, value in self.best_params.items():
                if key in ["risk_tolerance", "stop_loss_multiplier", "max_positions"]:
                    f.write(f"{key}: {value}\n")

    def generate_optimization_report(self) -> str:
        """
        Generate human-readable optimization report

        Returns:
            Report string
        """
        analysis = self.analyze_optimization_results()

        """
╔══════════════════════════════════════════════════════════╗
║     HYPERPARAMETER OPTIMIZATION REPORT                   ║
╚══════════════════════════════════════════════════════════╝

Target Metric: {target_metric}
Total Evaluations: {total_evals}
Best Score: {best_score:.4f}

═══ OPTIMAL PARAMETERS ═══
{optimal_params}

═══ SCORE STATISTICS ═══
Mean Score: {mean_score:.4f}
Std Dev: {std_score:.4f}
Min Score: {min_score:.4f}
Max Score: {max_score:.4f}

═══ PARAMETER IMPORTANCE ═══
{param_importance}

═══ CONVERGENCE ANALYSIS ═══
Converged: {converged}
Iterations to Best: {iter_to_best}
Improvement Rate: {improvement_rate:.6f}

═══ TOP 3 CONFIGURATIONS ═══
{top_configs}
        """.format(
            target_metric=self.config.target_metric.upper(),
            total_evals=analysis.get("total_evaluations", 0),
            best_score=self.best_score,
            optimal_params=self._format_params(self.best_params),
            mean_score=analysis["score_statistics"]["mean"],
            std_score=analysis["score_statistics"]["std"],
            min_score=analysis["score_statistics"]["min"],
            max_score=analysis["score_statistics"]["max"],
            param_importance=self._format_importance(
                analysis.get("parameter_importance", {})
            ),
            converged=analysis.get("convergence", {}).get("converged", "N/A"),
            iter_to_best=analysis.get("convergence", {}).get(
                "iterations_to_best", "N/A"
            ),
            improvement_rate=analysis.get("convergence", {}).get(
                "final_improvement_rate", 0
            ),
            top_configs=self._format_top_configs(
                analysis.get("top_5_configurations", [])[:3]
            ),
        )

        return report

    def _format_params(self, params: Dict) -> str:
        """Format parameters for display"""
        lines = []
        for key, value in params.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _format_importance(self, importance: Dict) -> str:
        """Format parameter importance"""
        lines = []
        for param, score in list(importance.items())[:5]:
            lines.append(f"  {param}: {'█' * int(score * 20):.20} {score:.3f}")
        return "\n".join(lines)

    def _format_top_configs(self, configs: List[Dict]) -> str:
        """Format top configurations"""
        lines = []
        for i, config in enumerate(configs, 1):
            lines.append(f"\nConfiguration {i} (Score: {config['score']:.4f}):")
            # Show only key parameters
            params = config["params"]
            key_params = ["model_weight_lstm", "model_weight_xgb", "risk_tolerance"]
            for param in key_params:
                if param in params:
                    lines.append(f"  {param}: {params[param]:.4f}")
        return "\n".join(lines)


async def main():
    """Main execution for testing"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION SYSTEM")
    print("Cloud Quant - Task Q-701")
    print("=" * 70)

    # Configure optimization
    config = OptimizationConfig(
        target_metric="sharpe_ratio",
        n_iterations=20,
        save_results=True,  # Reduced for demo
    )

    # Initialize tuner
    tuner = HyperparameterTuner(config)

    # Generate sample data for optimization
    print("\nGenerating sample data for optimization...")
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")

    sample_data = {}
    for symbol in ["AAPL", "GOOGL", "MSFT"]:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        df = pd.DataFrame(
            {"close": prices, "returns": pd.Series(prices).pct_change().fillna(0)},
            index=dates,
        )
        sample_data[symbol] = df

    # Run optimization
    print("\nStarting hyperparameter optimization...")
    print("This will take a few minutes...")

    await tuner.optimize_ml_parameters(sample_data)

    # Generate report
    tuner.generate_optimization_report()
    print(report)

    # Save results
    tuner.save_optimization_results()
    print("\nOptimization results saved to reports/optimal_parameters.yaml")

    return best_params


if __name__ == "__main__":
    asyncio.run(main())
