"""
Custom callbacks for RL training
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """
    Callback for tracking trading-specific metrics during training
    """

    def __init__(
        self, log_dir: Optional[Path] = None, log_freq: int = 1000, verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq

        # Metrics storage
        self.episode_metrics = []
        self.step_metrics = []
        self.trade_history = []

        # Current episode tracking
        self.current_episode_trades = 0
        self.current_episode_return = 0
        self.episode_start_value = None

    def _on_training_start(self) -> None:
        """Called before first rollout start"""
        self.start_time = datetime.now()
        logger.info("Trading metrics callback initialized")

    def _on_step(self) -> bool:
        """Called after each environment step"""
        # Extract info from environments
        for i, info in enumerate(self.locals.get("infos", [])):
            if info:
                # Track trades
                if "trade_details" in info and info["trade_details"]["shares"] != 0:
                    self.current_episode_trades += 1
                    self.trade_history.append(
                        {
                            "timestep": self.num_timesteps,
                            "episode": len(self.episode_metrics),
                            "action": info["trade_details"]["action"],
                            "shares": info["trade_details"]["shares"],
                            "price": info.get("current_price", 0),
                            "portfolio_value": info.get("portfolio_value", 0),
                        }
                    )

                # Initialize episode tracking
                if self.episode_start_value is None and "portfolio_value" in info:
                    self.episode_start_value = info["portfolio_value"]

                # Check for episode end
                if self.locals.get("dones", [False])[i]:
                    if self.episode_start_value and "portfolio_value" in info:
                        episode_return = (
                            info["portfolio_value"] - self.episode_start_value
                        ) / self.episode_start_value
                    else:
                        episode_return = 0

                    # Store episode metrics
                    episode_data = {
                        "episode": len(self.episode_metrics),
                        "timestep": self.num_timesteps,
                        "total_return": episode_return,
                        "n_trades": self.current_episode_trades,
                        "final_value": info.get("portfolio_value", 0),
                        "sharpe_ratio": info.get("sharpe_ratio", 0),
                        "max_drawdown": info.get("max_drawdown", 0),
                        "win_rate": info.get("win_rate", 0),
                    }
                    self.episode_metrics.append(episode_data)

                    # Reset episode tracking
                    self.current_episode_trades = 0
                    self.current_episode_return = 0
                    self.episode_start_value = None

                    # Log to tensorboard
                    if self.logger:
                        self.logger.record("trading/episode_return", episode_return)
                        self.logger.record("trading/n_trades", episode_data["n_trades"])
                        self.logger.record(
                            "trading/sharpe_ratio", episode_data["sharpe_ratio"]
                        )

        # Periodic logging
        if self.n_calls % self.log_freq == 0 and len(self.episode_metrics) > 0:
            self._log_statistics()

        return True

    def _log_statistics(self):
        """Log current statistics"""
        recent_episodes = self.episode_metrics[-10:]

        if recent_episodes:
            avg_return = np.mean([ep["total_return"] for ep in recent_episodes])
            avg_trades = np.mean([ep["n_trades"] for ep in recent_episodes])
            avg_sharpe = np.mean([ep["sharpe_ratio"] for ep in recent_episodes])

            if self.verbose > 0:
                logger.info(
                    f"Recent performance - Return: {avg_return:.2%}, "
                    f"Trades: {avg_trades:.1f}, Sharpe: {avg_sharpe:.2f}"
                )

            if self.logger:
                self.logger.record("trading/avg_return_10ep", avg_return)
                self.logger.record("trading/avg_trades_10ep", avg_trades)
                self.logger.record("trading/avg_sharpe_10ep", avg_sharpe)

    def _on_training_end(self) -> None:
        """Called at the end of training"""
        # Save all metrics
        metrics_data = {
            "episode_metrics": self.episode_metrics,
            "trade_history": self.trade_history[-1000:],  # Last 1000 trades
            "training_duration": (datetime.now() - self.start_time).total_seconds(),
            "total_episodes": len(self.episode_metrics),
            "total_trades": len(self.trade_history),
        }

        metrics_path = self.log_dir / "trading_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        logger.info(f"Saved trading metrics to {metrics_path}")


class PortfolioCallback(BaseCallback):
    """
    Callback for tracking portfolio state during training
    """

    def __init__(
        self, save_path: Optional[Path] = None, save_freq: int = 10000, verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path) if save_path else Path("./portfolio_history")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq

        # Portfolio tracking
        self.portfolio_snapshots = []
        self.position_history = []

    def _on_step(self) -> bool:
        """Called after each environment step"""
        # Track portfolio state
        for info in self.locals.get("infos", []):
            if info and "portfolio_value" in info:
                snapshot = {
                    "timestep": self.num_timesteps,
                    "portfolio_value": info["portfolio_value"],
                    "cash": info.get("cash", 0),
                    "position": info.get("position", 0),
                    "unrealized_pnl": info.get("unrealized_pnl", 0),
                    "realized_pnl": info.get("realized_pnl", 0),
                }
                self.portfolio_snapshots.append(snapshot)

                # Track position changes
                if info.get("position", 0) != 0:
                    self.position_history.append(
                        {
                            "timestep": self.num_timesteps,
                            "position": info["position"],
                            "price": info.get("current_price", 0),
                        }
                    )

        # Periodic save
        if self.n_calls % self.save_freq == 0:
            self._save_portfolio_history()

        return True

    def _save_portfolio_history(self):
        """Save portfolio history to file"""
        if not self.portfolio_snapshots:
            return

        # Convert to DataFrame for analysis
        portfolio_df = pd.DataFrame(self.portfolio_snapshots)

        # Calculate additional metrics
        portfolio_df["returns"] = portfolio_df["portfolio_value"].pct_change()
        portfolio_df["cumulative_returns"] = (1 + portfolio_df["returns"]).cumprod() - 1

        # Save to CSV
        csv_path = self.save_path / f"portfolio_history_{self.num_timesteps}.csv"
        portfolio_df.to_csv(csv_path, index=False)

        if self.verbose > 0:
            logger.info(f"Saved portfolio history to {csv_path}")

    def _on_training_end(self) -> None:
        """Called at the end of training"""
        self._save_portfolio_history()

        # Save position history
        if self.position_history:
            position_df = pd.DataFrame(self.position_history)
            position_path = self.save_path / "position_history.csv"
            position_df.to_csv(position_path, index=False)


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping based on validation performance
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        patience: int = 10,
        min_improvement: float = 0.01,
        metric: str = "sharpe_ratio",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric

        self.best_score = -np.inf
        self.patience_counter = 0
        self.eval_history = []

    def _on_step(self) -> bool:
        """Check for early stopping"""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            eval_score = self._evaluate_model()
            self.eval_history.append(
                {"timestep": self.num_timesteps, "score": eval_score}
            )

            # Check for improvement
            if eval_score > self.best_score + self.min_improvement:
                self.best_score = eval_score
                self.patience_counter = 0

                if self.verbose > 0:
                    logger.info(f"New best {self.metric}: {eval_score:.4f}")
            else:
                self.patience_counter += 1

                if self.verbose > 0:
                    logger.info(
                        f"No improvement. Patience: {self.patience_counter}/{self.patience}"
                    )

            # Check early stopping condition
            if self.patience_counter >= self.patience:
                if self.verbose > 0:
                    logger.info("Early stopping triggered!")
                return False  # Stop training

        return True  # Continue training

    def _evaluate_model(self) -> float:
        """Evaluate model on validation environment"""
        episode_scores = []

        for _ in range(self.n_eval_episodes):
            self.eval_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.eval_env.step(action)

            # Extract metric
            if hasattr(self.eval_env, "get_episode_summary"):
                summary = self.eval_env.get_episode_summary()
                score = summary.get(self.metric, 0)
            else:
                score = info.get(self.metric, 0)

            episode_scores.append(score)

        return np.mean(episode_scores)


class CheckpointCallback(BaseCallback):
    """
    Save model checkpoints during training
    """

    def __init__(
        self,
        save_path: Path,
        save_freq: int = 50000,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer

    def _on_step(self) -> bool:
        """Save checkpoint if needed"""
        if self.n_calls % self.save_freq == 0:
            # Create checkpoint name
            checkpoint_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            checkpoint_path = self.save_path / checkpoint_name

            # Save model
            self.model.save(checkpoint_path)

            # Save replay buffer if applicable
            if self.save_replay_buffer and hasattr(self.model, "replay_buffer"):
                buffer_path = self.save_path / f"{checkpoint_name}_buffer"
                self.model.save_replay_buffer(buffer_path)

            if self.verbose > 0:
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        return True
