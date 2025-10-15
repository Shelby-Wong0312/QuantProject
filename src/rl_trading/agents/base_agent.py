"""
Base class for RL trading agents
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for RL trading agents
    """

    def __init__(self, env, config: Dict[str, Any], name: str = "BaseAgent"):
        """
        Initialize base agent

        Args:
            env: Trading environment
            config: Agent configuration
            name: Agent name
        """
        self.env = env
        self.config = config
        self.name = name

        # Training metrics
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_metrics": [],
            "learning_progress": [],
        }

        # Best model tracking
        self.best_reward = -np.inf
        self.best_model_path = None

        logger.info(f"Initialized {name} agent")

    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        eval_env: Optional[Any] = None,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ) -> None:
        """
        Train the agent

        Args:
            total_timesteps: Total training timesteps
            callback: Training callback
            log_interval: Logging interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Whether to reset timestep counter
        """
        pass

    @abstractmethod
    def predict(
        self, observation: np.ndarray, state: Optional[Any] = None, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predict action given observation

        Args:
            observation: Current observation
            state: Agent state
            deterministic: Use deterministic policy

        Returns:
            Action and next state
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save agent model

        Args:
            path: Save path
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load agent model

        Args:
            path: Load path
        """
        pass

    def evaluate(
        self, eval_env, n_episodes: int = 10, deterministic: bool = True, render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate agent performance

        Args:
            eval_env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            render: Whether to render environment

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []

        for episode in range(n_episodes):
            eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)

                episode_reward += reward
                episode_length += 1

                if render:
                    eval_env.render()

            # Get episode summary
            if hasattr(eval_env, "get_episode_summary"):
                episode_summary = eval_env.get_episode_summary()
                episode_metrics.append(episode_summary)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Reward = {episode_reward:.2f}, Length = {episode_length}"
            )

        # Calculate aggregated metrics
        metrics = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "total_episodes": n_episodes,
        }

        # Add trading-specific metrics if available
        if episode_metrics:
            trading_metrics = self._aggregate_trading_metrics(episode_metrics)
            metrics.update(trading_metrics)

        return metrics

    def _aggregate_trading_metrics(self, episode_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate trading-specific metrics"""
        aggregated = {}

        # Common trading metrics
        metric_keys = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "avg_trade_size",
        ]

        for key in metric_keys:
            values = [m.get(key, 0) for m in episode_metrics if key in m]
            if values:
                aggregated[f"mean_{key}"] = np.mean(values)
                aggregated[f"std_{key}"] = np.std(values)

        return aggregated

    def save_training_history(self, path: Union[str, Path]) -> None:
        """
        Save training history

        Args:
            path: Save path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        history_data = {
            "agent_name": self.name,
            "config": self.config,
            "training_history": self.training_history,
            "best_reward": float(self.best_reward),
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
        }

        with open(path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Saved training history to {path}")

    def load_training_history(self, path: Union[str, Path]) -> None:
        """
        Load training history

        Args:
            path: Load path
        """
        path = Path(path)

        with open(path, "r") as f:
            history_data = json.load(f)

        self.training_history = history_data["training_history"]
        self.best_reward = history_data["best_reward"]
        self.best_model_path = (
            Path(history_data["best_model_path"]) if history_data["best_model_path"] else None
        )

        logger.info(f"Loaded training history from {path}")

    def get_action_distribution(
        self, observations: np.ndarray, n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Get action distribution for given observations

        Args:
            observations: Observations
            n_samples: Number of samples

        Returns:
            Action distribution statistics
        """
        actions = []

        for _ in range(n_samples):
            action, _ = self.predict(observations, deterministic=False)
            actions.append(action)

        actions = np.array(actions)

        return {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
            "min": np.min(actions, axis=0),
            "max": np.max(actions, axis=0),
            "median": np.median(actions, axis=0),
        }

    def analyze_performance(self, test_env, n_episodes: int = 20) -> pd.DataFrame:
        """
        Detailed performance analysis

        Args:
            test_env: Test environment
            n_episodes: Number of test episodes

        Returns:
            Performance analysis DataFrame
        """
        results = []

        for episode in range(n_episodes):
            test_env.reset()
            done = False

            episode_data = {
                "episode": episode,
                "actions": [],
                "rewards": [],
                "positions": [],
                "portfolio_values": [],
            }

            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)

                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["positions"].append(info.get("position", 0))
                episode_data["portfolio_values"].append(info.get("portfolio_value", 0))

            # Calculate episode metrics
            episode_summary = test_env.get_episode_summary()
            episode_data.update(episode_summary)

            results.append(episode_data)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)

        # Add additional analysis
        df["cumulative_reward"] = df["rewards"].apply(lambda x: np.cumsum(x)[-1] if x else 0)
        df["max_position"] = df["positions"].apply(lambda x: max(x) if x else 0)
        df["position_changes"] = df["positions"].apply(
            lambda x: sum(1 for i in range(1, len(x)) if x[i] != x[i - 1]) if len(x) > 1 else 0
        )

        return df
