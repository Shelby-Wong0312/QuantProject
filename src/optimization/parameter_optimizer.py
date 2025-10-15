"""
Parameter Optimization System
Advanced parameter optimization using Optuna/Scikit-Optimize
Stage 8 - Strategy Optimization Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Advanced parameter optimization using Optuna for hyperparameter tuning
    Supports multiple optimization algorithms and parallel execution
    """

    def __init__(
        self,
        study_name: str = "strategy_optimization",
        direction: str = "maximize",
        sampler: str = "tpe",
        pruner: str = "median",
        n_jobs: int = -1,
    ):
        """
        Initialize parameter optimizer

        Args:
            study_name: Name of the optimization study
            direction: "maximize" or "minimize"
            sampler: "tpe", "cmaes", or "random"
            pruner: "median", "hyperband", or None
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.study_name = study_name
        self.direction = direction
        self.n_jobs = n_jobs

        # Initialize sampler
        if sampler == "tpe":
            self.sampler = TPESampler(n_startup_trials=20, n_ei_candidates=24)
        elif sampler == "cmaes":
            self.sampler = CmaEsSampler()
        else:
            self.sampler = optuna.samplers.RandomSampler()

        # Initialize pruner
        if pruner == "median":
            self.pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        elif pruner == "hyperband":
            self.pruner = HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
        else:
            self.pruner = None

        # Create study
        self.study = optuna.create_study(
            study_name=f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )

        self.optimization_history = []
        self.best_params = None
        self.best_value = None

        logger.info(f"Parameter optimizer initialized: {sampler} sampler, {pruner} pruner")

    def grid_search(
        self, strategy: Any, param_grid: Dict[str, List], objective_func: Callable, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform grid search optimization

        Args:
            strategy: Trading strategy instance
            param_grid: Parameter grid to search
            objective_func: Objective function to optimize
            **kwargs: Additional arguments for objective function

        Returns:
            Best parameters and optimization results
        """
        logger.info(f"Starting grid search with {len(param_grid)} parameters")

        # Generate all parameter combinations
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Total combinations to evaluate: {len(combinations)}")

        best_params = None
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        results = []

        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))

            try:
                score = objective_func(strategy, params, **kwargs)

                results.append({"params": params, "score": score, "iteration": i + 1})

                # Update best
                if self.direction == "maximize":
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Grid search progress: {i+1}/{len(combinations)}, "
                        f"Best score: {best_score:.4f}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating combination {i+1}: {e}")
                continue

        self.best_params = best_params
        self.best_value = best_score
        self.optimization_history = results

        logger.info(f"Grid search completed. Best score: {best_score:.4f}")

        return {
            "best_params": best_params,
            "best_value": best_score,
            "n_trials": len(combinations),
            "optimization_history": results,
        }

    def bayesian_optimization(
        self,
        strategy: Any,
        param_bounds: Dict[str, Union[Tuple, List]],
        objective_func: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna

        Args:
            strategy: Trading strategy instance
            param_bounds: Parameter bounds {param_name: (min, max) or [choices]}
            objective_func: Objective function to optimize
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            **kwargs: Additional arguments for objective function

        Returns:
            Best parameters and optimization results
        """
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")

        def objective(trial):
            """Optuna objective function"""
            params = {}

            for param_name, bounds in param_bounds.items():
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    # Continuous parameter
                    if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                        params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, bounds)
                else:
                    raise ValueError(f"Invalid bounds for parameter {param_name}: {bounds}")

            try:
                score = objective_func(strategy, params, **kwargs)

                # Report intermediate value for pruning
                trial.report(score, step=1)

                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                return float("-inf") if self.direction == "maximize" else float("inf")

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,  # Use single job to avoid conflicts
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Extract optimization history
        self.optimization_history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.optimization_history.append(
                    {"params": trial.params, "score": trial.value, "iteration": trial.number + 1}
                )

        logger.info(f"Bayesian optimization completed. Best score: {self.best_value:.4f}")

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "optimization_history": self.optimization_history,
            "study": self.study,
        }

    def genetic_algorithm(
        self,
        population_size: int,
        fitness_func: Callable,
        param_bounds: Dict[str, Tuple],
        n_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform genetic algorithm optimization

        Args:
            population_size: Size of population
            fitness_func: Fitness function to optimize
            param_bounds: Parameter bounds
            n_generations: Number of generations
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            **kwargs: Additional arguments for fitness function

        Returns:
            Best parameters and optimization results
        """
        logger.info(f"Starting genetic algorithm with population size {population_size}")

        param_names = list(param_bounds.keys())
        n_params = len(param_names)

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, (min_val, max_val) in param_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    individual[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    individual[param_name] = np.random.uniform(min_val, max_val)
            population.append(individual)

        best_individual = None
        best_fitness = float("-inf") if self.direction == "maximize" else float("inf")
        fitness_history = []

        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    fitness = fitness_func(individual, **kwargs)
                    fitness_scores.append(fitness)

                    # Update best
                    if self.direction == "maximize":
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_individual = individual.copy()
                    else:
                        if fitness < best_fitness:
                            best_fitness = fitness
                            best_individual = individual.copy()

                except Exception as e:
                    logger.error(f"Error evaluating individual in generation {generation}: {e}")
                    fitness_scores.append(
                        float("-inf") if self.direction == "maximize" else float("inf")
                    )

            fitness_history.append(
                {
                    "generation": generation + 1,
                    "best_fitness": best_fitness,
                    "avg_fitness": np.mean(fitness_scores),
                    "std_fitness": np.std(fitness_scores),
                }
            )

            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(
                    population_size, tournament_size, replace=False
                )
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]

                if self.direction == "maximize":
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                else:
                    winner_idx = tournament_indices[np.argmin(tournament_fitness)]

                new_population.append(population[winner_idx].copy())

            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                parent1 = new_population[i]
                parent2 = new_population[i + 1]

                # Crossover
                if np.random.random() < crossover_rate:
                    for param_name in param_names:
                        if np.random.random() < 0.5:
                            parent1[param_name], parent2[param_name] = (
                                parent2[param_name],
                                parent1[param_name],
                            )

                # Mutation
                for parent in [parent1, parent2]:
                    if np.random.random() < mutation_rate:
                        param_name = np.random.choice(param_names)
                        min_val, max_val = param_bounds[param_name]

                        if isinstance(min_val, int) and isinstance(max_val, int):
                            parent[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            parent[param_name] = np.random.uniform(min_val, max_val)

            population = new_population

            if (generation + 1) % 10 == 0:
                logger.info(
                    f"Generation {generation + 1}/{n_generations}, "
                    f"Best fitness: {best_fitness:.4f}, "
                    f"Avg fitness: {np.mean(fitness_scores):.4f}"
                )

        self.best_params = best_individual
        self.best_value = best_fitness
        self.optimization_history = fitness_history

        logger.info(f"Genetic algorithm completed. Best fitness: {best_fitness:.4f}")

        return {
            "best_params": best_individual,
            "best_value": best_fitness,
            "n_generations": n_generations,
            "optimization_history": fitness_history,
        }

    def multi_objective_optimization(
        self,
        strategy: Any,
        param_bounds: Dict[str, Union[Tuple, List]],
        objective_funcs: List[Callable],
        objective_names: List[str],
        n_trials: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform multi-objective optimization

        Args:
            strategy: Trading strategy instance
            param_bounds: Parameter bounds
            objective_funcs: List of objective functions
            objective_names: Names of objectives
            n_trials: Number of trials
            **kwargs: Additional arguments

        Returns:
            Pareto front and optimization results
        """
        logger.info(f"Starting multi-objective optimization with {len(objective_funcs)} objectives")

        # Create multi-objective study
        study = optuna.create_study(
            directions=["maximize"] * len(objective_funcs), sampler=self.sampler
        )

        def objective(trial):
            """Multi-objective function"""
            params = {}

            for param_name, bounds in param_bounds.items():
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                        params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    params[param_name] = trial.suggest_categorical(param_name, bounds)

            try:
                # Evaluate all objectives
                objective_values = []
                for obj_func in objective_funcs:
                    value = obj_func(strategy, params, **kwargs)
                    objective_values.append(value)

                return objective_values

            except Exception as e:
                logger.error(f"Error in multi-objective trial {trial.number}: {e}")
                return [float("-inf")] * len(objective_funcs)

        # Run optimization
        study.optimize(objective, n_trials=n_trials)

        # Extract Pareto front
        pareto_front = []
        for trial in study.best_trials:
            pareto_front.append(
                {
                    "params": trial.params,
                    "values": trial.values,
                    "objectives": dict(zip(objective_names, trial.values)),
                }
            )

        logger.info(
            f"Multi-objective optimization completed. Pareto front size: {len(pareto_front)}"
        )

        return {
            "pareto_front": pareto_front,
            "study": study,
            "n_trials": len(study.trials),
            "objective_names": objective_names,
        }

    def save_optimization_results(
        self, filepath: str = "reports/parameter_optimization_results.json"
    ):
        """
        Save optimization results to file

        Args:
            filepath: Path to save results
        """
        results = {
            "study_name": self.study_name,
            "optimization_config": {
                "direction": self.direction,
                "sampler": type(self.sampler).__name__,
                "pruner": type(self.pruner).__name__ if self.pruner else None,
                "timestamp": datetime.now().isoformat(),
            },
            "best_parameters": self.best_params,
            "best_value": self.best_value,
            "optimization_history": self.optimization_history[:100],  # Limit size
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Optimization results saved to {filepath}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get optimization summary

        Returns:
            Summary dictionary
        """
        if not self.optimization_history:
            return {}

        scores = [entry["score"] for entry in self.optimization_history]

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "total_trials": len(self.optimization_history),
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
            },
            "convergence_analysis": self._analyze_convergence(),
        }

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        if len(self.optimization_history) < 10:
            return {}

        scores = [entry["score"] for entry in self.optimization_history]

        # Calculate running best
        running_best = []
        current_best = float("-inf") if self.direction == "maximize" else float("inf")

        for score in scores:
            if self.direction == "maximize":
                current_best = max(current_best, score)
            else:
                current_best = min(current_best, score)
            running_best.append(current_best)

        # Check convergence
        n_check = max(1, len(scores) // 5)
        converged = all(running_best[-n_check:] == running_best[-1])

        return {
            "converged": converged,
            "iterations_to_best": (
                running_best.index(self.best_value) + 1
                if self.best_value in running_best
                else len(running_best)
            ),
            "improvement_rate": (
                (running_best[-1] - running_best[0]) / len(running_best)
                if len(running_best) > 1
                else 0
            ),
        }


def create_objective_function(
    backtest_engine: Any, data: Dict[str, pd.DataFrame], target_metric: str = "sharpe_ratio"
) -> Callable:
    """
    Create objective function for optimization

    Args:
        backtest_engine: Backtesting engine
        data: Historical data
        target_metric: Target metric to optimize

    Returns:
        Objective function
    """

    def objective(strategy: Any, params: Dict[str, Any], **kwargs) -> float:
        """
        Objective function to maximize/minimize

        Args:
            strategy: Strategy instance
            params: Parameters to evaluate
            **kwargs: Additional arguments

        Returns:
            Objective value
        """
        try:
            # Update strategy parameters
            for param_name, param_value in params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
                elif hasattr(strategy.config, param_name):
                    setattr(strategy.config, param_name, param_value)
                elif hasattr(strategy.config, "parameters"):
                    strategy.config.parameters[param_name] = param_value

            # Run backtest
            results = backtest_engine.run_backtest(strategy, data)

            # Return target metric
            if target_metric == "sharpe_ratio":
                return results.get("sharpe_ratio", 0)
            elif target_metric == "annual_return":
                return results.get("annual_return", 0)
            elif target_metric == "calmar_ratio":
                return results.get("calmar_ratio", 0)
            elif target_metric == "profit_factor":
                return results.get("profit_factor", 0)
            elif target_metric == "win_rate":
                return results.get("win_rate", 0)
            else:
                return results.get("total_return", 0)

        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return float("-inf")

    return objective


async def main():
    """Test parameter optimization"""
    print("\n" + "=" * 70)
    print("PARAMETER OPTIMIZATION SYSTEM")
    print("Stage 8 - Strategy Optimization Framework")
    print("=" * 70)

    # Initialize optimizer
    optimizer = ParameterOptimizer(
        study_name="test_optimization", direction="maximize", sampler="tpe"
    )

    # Test grid search
    print("\n1. Testing Grid Search...")
    param_grid = {
        "ma_short": [10, 20, 30],
        "ma_long": [50, 100, 200],
        "threshold": [0.01, 0.02, 0.03],
    }

    def simple_objective(strategy, params):
        # Simplified objective for testing
        return np.random.random() + params["ma_short"] / 100

    grid_results = optimizer.grid_search(
        strategy=None, param_grid=param_grid, objective_func=simple_objective
    )

    print(f"Grid search completed. Best score: {grid_results['best_value']:.4f}")
    print(f"Best parameters: {grid_results['best_params']}")

    # Test Bayesian optimization
    print("\n2. Testing Bayesian Optimization...")
    param_bounds = {"ma_short": (5, 50), "ma_long": (20, 200), "threshold": (0.005, 0.05)}

    bayes_results = optimizer.bayesian_optimization(
        strategy=None, param_bounds=param_bounds, objective_func=simple_objective, n_trials=50
    )

    print(f"Bayesian optimization completed. Best score: {bayes_results['best_value']:.4f}")
    print(f"Best parameters: {bayes_results['best_params']}")

    # Save results
    optimizer.save_optimization_results()

    # Print summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Best value: {summary['best_value']:.4f}")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Converged: {summary.get('convergence_analysis', {}).get('converged', 'Unknown')}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
