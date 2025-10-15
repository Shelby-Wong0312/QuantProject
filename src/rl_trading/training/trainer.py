"""
Main trainer for RL trading agents
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from rl_trading.environments import TradingEnvironment
from rl_trading.agents import PPOAgent
from rl_trading.configs.env_config import get_config, TradingEnvConfig
from .evaluation import Evaluator
from .callbacks import TradingMetricsCallback, PortfolioCallback

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for RL trading agents
    """

    def __init__(
        self,
        agent_type: str = "PPO",
        env_config: Optional[TradingEnvConfig] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
        results_dir: Union[str, Path] = "./results",
        seed: Optional[int] = None,
    ):
        """
        Initialize trainer

        Args:
            agent_type: Type of agent ('PPO', 'SAC', etc.)
            env_config: Environment configuration
            agent_config: Agent configuration
            experiment_name: Name for this experiment
            results_dir: Directory for results
            seed: Random seed
        """
        self.agent_type = agent_type
        self.env_config = env_config or get_config("day_trading")
        self.agent_config = agent_config or {}

        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{agent_type}_{timestamp}"
        self.experiment_name = experiment_name

        # Setup directories
        self.results_dir = Path(results_dir) / experiment_name
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.plots_dir = self.results_dir / "plots"

        for dir_path in [self.models_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set seed
        if seed is not None:
            self.seed = seed
            set_random_seed(seed)
        else:
            self.seed = np.random.randint(0, 10000)

        # Initialize environments
        self.train_env = None
        self.val_env = None
        self.test_env = None
        self.agent = None

        # Training history
        self.training_history = {
            "training_rewards": [],
            "validation_scores": [],
            "best_val_score": -np.inf,
            "training_time": 0,
        }

        logger.info(f"Initialized trainer for experiment: {experiment_name}")

    def setup_environments(
        self,
        train_data_path: Optional[Path] = None,
        val_data_path: Optional[Path] = None,
        test_data_path: Optional[Path] = None,
    ):
        """
        Setup training, validation, and test environments

        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            test_data_path: Path to test data
        """
        logger.info("Setting up environments...")

        # Create training environment
        self.train_env = self._create_environment(data_path=train_data_path, is_training=True)

        # Create validation environment
        if val_data_path:
            self.val_env = self._create_environment(data_path=val_data_path, is_training=False)

        # Create test environment
        if test_data_path:
            self.test_env = self._create_environment(data_path=test_data_path, is_training=False)

        logger.info("Environments setup completed")

    def _create_environment(
        self, data_path: Optional[Path] = None, is_training: bool = True
    ) -> TradingEnvironment:
        """Create a trading environment"""
        env = TradingEnvironment(
            symbol=self.env_config.symbols[0],
            data_path=data_path,
            initial_capital=self.env_config.initial_capital,
            max_steps=self.env_config.max_steps_per_episode,
            window_size=self.env_config.window_size,
            action_type=self.env_config.action_type,
            random_start=is_training and self.env_config.random_start,
            seed=self.seed if is_training else None,
        )

        return env

    def setup_agent(self):
        """Setup RL agent"""
        logger.info(f"Setting up {self.agent_type} agent...")

        if self.train_env is None:
            raise ValueError("Must setup environments before agent")

        # Update agent config with experiment settings
        self.agent_config.update(
            {"tensorboard_log": str(self.logs_dir / "tensorboard"), "seed": self.seed}
        )

        # Create agent based on type
        if self.agent_type == "PPO":
            self.agent = PPOAgent(
                env=self.train_env, config=self.agent_config, name=f"PPO_{self.experiment_name}"
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        logger.info("Agent setup completed")

    def train(
        self,
        total_timesteps: int = 1000000,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        n_eval_episodes: int = 5,
        early_stopping_patience: int = 10,
        verbose: int = 1,
    ):
        """
        Train the agent

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_freq: Model save frequency
            n_eval_episodes: Number of evaluation episodes
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level
        """
        if self.agent is None:
            raise ValueError("Must setup agent before training")

        logger.info(f"Starting training for {total_timesteps} timesteps")
        start_time = datetime.now()

        # Setup callbacks
        callbacks = []

        # Trading metrics callback
        metrics_callback = TradingMetricsCallback(log_dir=self.logs_dir, verbose=verbose)
        callbacks.append(metrics_callback)

        # Portfolio tracking callback
        portfolio_callback = PortfolioCallback(
            save_path=self.logs_dir / "portfolio", verbose=verbose
        )
        callbacks.append(portfolio_callback)

        # Train agent
        self.agent.train(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            eval_env=self.val_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_freq=save_freq,
            save_path=self.models_dir,
        )

        # Calculate training time
        end_time = datetime.now()
        self.training_history["training_time"] = (end_time - start_time).total_seconds()

        logger.info(f"Training completed in {self.training_history['training_time']:.2f} seconds")

        # Save final model
        self.save_model("final_model")

        # Save training history
        self.save_training_history()

        # Generate training report
        self.generate_training_report()

    def evaluate(
        self, env_type: str = "test", n_episodes: int = 20, save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agent

        Args:
            env_type: Environment to evaluate on ('train', 'val', 'test')
            n_episodes: Number of evaluation episodes
            save_results: Whether to save results

        Returns:
            Evaluation results
        """
        # Select environment
        env_map = {"train": self.train_env, "val": self.val_env, "test": self.test_env}

        eval_env = env_map.get(env_type)
        if eval_env is None:
            raise ValueError(f"No {env_type} environment available")

        # Create evaluator
        evaluator = Evaluator(
            agent=self.agent, results_dir=self.results_dir / f"{env_type}_evaluation"
        )

        # Run evaluation
        results = evaluator.evaluate(env=eval_env, n_episodes=n_episodes, save_results=save_results)

        return results

    def save_model(self, name: str = "model"):
        """Save the current model"""
        model_path = self.models_dir / name
        self.agent.save(model_path)
        logger.info(f"Saved model to {model_path}")

    def load_model(self, name: str = "model"):
        """Load a saved model"""
        model_path = self.models_dir / name
        self.agent.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def save_training_history(self):
        """Save training history"""
        history_path = self.results_dir / "training_history.json"

        # Add experiment metadata
        history_data = {
            "experiment_name": self.experiment_name,
            "agent_type": self.agent_type,
            "env_config": self.env_config.__dict__,
            "agent_config": self._serialize_config(self.agent_config),
            "seed": self.seed,
            "training_history": self.training_history,
        }

        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2, default=str)

        logger.info(f"Saved training history to {history_path}")

    def _serialize_config(self, config: Dict) -> Dict:
        """Serialize config for JSON"""
        serialized = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    def generate_training_report(self):
        """Generate comprehensive training report"""
        report_path = self.results_dir / "training_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"TRAINING REPORT - {self.experiment_name}\n")
            f.write("=" * 60 + "\n\n")

            # Experiment details
            f.write("EXPERIMENT DETAILS:\n")
            f.write(f"  Agent Type: {self.agent_type}\n")
            f.write(f"  Symbols: {self.env_config.symbols}\n")
            f.write(f"  Initial Capital: ${self.env_config.initial_capital:,}\n")
            f.write(f"  Episode Length: {self.env_config.max_steps_per_episode}\n")
            f.write(f"  Random Seed: {self.seed}\n")
            f.write(f"  Training Time: {self.training_history['training_time']:.2f} seconds\n")

            # Training metrics
            if self.agent:
                metrics = self.agent.get_training_metrics()
                f.write("\nTRAINING METRICS:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")

            # Best validation score
            f.write(f"\nBest Validation Score: {self.training_history['best_val_score']:.4f}\n")

            # Model details
            if self.agent:
                policy_stats = self.agent.get_policy_stats()
                f.write("\nMODEL DETAILS:\n")
                for key, value in policy_stats.items():
                    f.write(f"  {key}: {value}\n")

        logger.info(f"Generated training report at {report_path}")

        # Generate plots
        self._generate_training_plots()

    def _generate_training_plots(self):
        """Generate training visualization plots"""
        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")

        # Plot 1: Training rewards
        if self.agent and self.agent.training_history["episode_rewards"]:
            fig, ax = plt.subplots(figsize=(10, 6))

            rewards = self.agent.training_history["episode_rewards"]
            episodes = range(len(rewards))

            # Plot raw rewards
            ax.plot(episodes, rewards, alpha=0.3, label="Episode Rewards")

            # Plot moving average
            window = min(100, len(rewards) // 10)
            if window > 1:
                ma = pd.Series(rewards).rolling(window).mean()
                ax.plot(episodes, ma, label=f"{window}-Episode MA", linewidth=2)

            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title("Training Rewards Over Time")
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.plots_dir / "training_rewards.png")
            plt.close()

        # Plot 2: Portfolio metrics
        if hasattr(self, "portfolio_history") and self.portfolio_history:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Portfolio value
            ax = axes[0, 0]
            values = [h["portfolio_value"] for h in self.portfolio_history]
            ax.plot(values)
            ax.set_title("Portfolio Value")
            ax.set_ylabel("Value ($)")

            # Returns
            ax = axes[0, 1]
            returns = pd.Series(values).pct_change().dropna()
            ax.hist(returns, bins=50, alpha=0.7)
            ax.set_title("Return Distribution")
            ax.set_xlabel("Returns")

            # Position sizes
            ax = axes[1, 0]
            positions = [h.get("position", 0) for h in self.portfolio_history]
            ax.plot(positions)
            ax.set_title("Position Over Time")
            ax.set_ylabel("Shares")

            # Drawdown
            ax = axes[1, 1]
            peak = pd.Series(values).expanding().max()
            drawdown = (pd.Series(values) - peak) / peak
            ax.fill_between(range(len(drawdown)), drawdown, alpha=0.7, color="red")
            ax.set_title("Drawdown")
            ax.set_ylabel("Drawdown %")

            plt.tight_layout()
            plt.savefig(self.plots_dir / "portfolio_metrics.png")
            plt.close()

        logger.info("Generated training plots")

    def run_hyperparameter_search(
        self,
        param_space: Dict[str, List[Any]],
        n_trials: int = 20,
        n_timesteps_per_trial: int = 100000,
    ):
        """
        Run hyperparameter search

        Args:
            param_space: Dictionary of parameter ranges
            n_trials: Number of trials
            n_timesteps_per_trial: Timesteps per trial
        """
        logger.info(f"Starting hyperparameter search with {n_trials} trials")

        # Import Optuna if available
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            return

        def objective(trial):
            # Sample hyperparameters
            config = {}
            for param, values in param_space.items():
                if isinstance(values, list):
                    config[param] = trial.suggest_categorical(param, values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], int):
                        config[param] = trial.suggest_int(param, values[0], values[1])
                    else:
                        config[param] = trial.suggest_float(param, values[0], values[1])

            # Update agent config
            self.agent_config.update(config)

            # Setup new agent
            self.setup_agent()

            # Train
            self.agent.train(
                total_timesteps=n_timesteps_per_trial,
                eval_env=self.val_env,
                eval_freq=10000,
                n_eval_episodes=3,
            )

            # Evaluate
            results = self.evaluate("val", n_episodes=5, save_results=False)

            # Return metric to optimize (e.g., Sharpe ratio)
            return results.get("mean_sharpe_ratio", 0)

        # Create study
        study = optuna.create_study(
            direction="maximize", study_name=f"hyperparam_search_{self.experiment_name}"
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials)

        # Save results
        results_path = self.results_dir / "hyperparameter_search_results.json"

        best_params = study.best_params
        best_value = study.best_value

        search_results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_trials,
            "all_trials": [
                {"params": trial.params, "value": trial.value} for trial in study.trials
            ],
        }

        with open(results_path, "w") as f:
            json.dump(search_results, f, indent=2)

        logger.info(f"Hyperparameter search completed. Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params
