"""
Portfolio Management Agent using PPO for Multi-Asset Trading
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PortfolioFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for portfolio management
    Handles multi-asset features and portfolio state
    """

    def __init__(
        self, observation_space, n_assets: int, features_per_asset: int = 20, hidden_dim: int = 256
    ):
        # Output dimension after feature extraction
        super().__init__(observation_space, features_dim=hidden_dim)

        self.n_assets = n_assets
        self.features_per_asset = features_per_asset

        # Asset feature processing
        self.asset_encoder = nn.Sequential(
            nn.Linear(features_per_asset, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Attention mechanism for assets
        self.asset_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.1)

        # Portfolio state processing
        portfolio_features = observation_space.shape[0] - (n_assets * features_per_asset)
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )

        # Combine features
        self.feature_combiner = nn.Sequential(
            nn.Linear(n_assets * 64 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split observations into asset features and portfolio features
        asset_features_flat = observations[:, : self.n_assets * self.features_per_asset]
        portfolio_features = observations[:, self.n_assets * self.features_per_asset :]

        # Reshape asset features: (batch, n_assets, features_per_asset)
        asset_features = asset_features_flat.view(
            batch_size, self.n_assets, self.features_per_asset
        )

        # Process each asset's features
        asset_embeddings = []
        for i in range(self.n_assets):
            asset_embed = self.asset_encoder(asset_features[:, i, :])
            asset_embeddings.append(asset_embed)

        # Stack embeddings: (n_assets, batch, embed_dim)
        asset_embeddings = torch.stack(asset_embeddings, dim=0)

        # Apply attention
        attended_assets, _ = self.asset_attention(
            asset_embeddings, asset_embeddings, asset_embeddings
        )

        # Flatten attended assets: (batch, n_assets * embed_dim)
        attended_assets = attended_assets.transpose(0, 1).contiguous()
        attended_assets = attended_assets.view(batch_size, -1)

        # Process portfolio features
        portfolio_embed = self.portfolio_encoder(portfolio_features)

        # Combine all features
        combined = torch.cat([attended_assets, portfolio_embed], dim=1)
        features = self.feature_combiner(combined)

        return features


class PortfolioPolicy(ActorCriticPolicy):
    """
    Custom policy for portfolio management
    Outputs weight allocations that sum to 1
    """

    def __init__(self, *args, n_assets: int = 5, **kwargs):
        self.n_assets = n_assets
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)

        # Override action distribution to ensure valid portfolio weights
        # The action net already outputs the right dimension
        # We'll handle normalization in post-processing


class PortfolioAgent:
    """
    Portfolio management agent using PPO
    """

    def __init__(self, env, n_assets: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize portfolio agent

        Args:
            env: Portfolio trading environment
            n_assets: Number of assets to trade
            config: Agent configuration
        """
        self.env = env
        self.n_assets = n_assets

        # Default configuration
        default_config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "hidden_dim": 256,
            "features_per_asset": 20,
        }

        self.config = default_config
        if config:
            self.config.update(config)

        # Create PPO model with custom feature extractor
        policy_kwargs = {
            "features_extractor_class": PortfolioFeatureExtractor,
            "features_extractor_kwargs": {
                "n_assets": n_assets,
                "features_per_asset": self.config["features_per_asset"],
                "hidden_dim": self.config["hidden_dim"],
            },
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "activation_fn": nn.ReLU,
        }

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/portfolio_ppo/",
            verbose=1,
        )

        logger.info(f"Initialized portfolio agent for {n_assets} assets")

    def train(
        self,
        total_timesteps: int = 1000000,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10000,
        save_path: str = "./models/portfolio_ppo",
        callback_list: Optional[List] = None,
    ):
        """
        Train the portfolio agent

        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency
            save_path: Path to save models
            callback_list: Additional callbacks
        """
        callbacks = callback_list or []

        # Add evaluation callback
        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=f"{save_path}/logs",
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        # Add checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=50000, save_path=f"{save_path}/checkpoints", name_prefix="portfolio_model"
        )
        callbacks.append(checkpoint_callback)

        # Train model
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Save final model
        self.save(f"{save_path}/final_model")
        logger.info("Training completed")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict portfolio weights

        Args:
            observation: Current state observation
            deterministic: Use deterministic policy

        Returns:
            action: Portfolio weights (normalized)
            states: RNN states (if applicable)
        """
        # Get raw action from model
        action, states = self.model.predict(observation, deterministic=deterministic)

        # Ensure action is valid portfolio weights
        action = self._normalize_weights(action)

        return action, states

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1 and be non-negative"""
        # Apply softmax to ensure weights sum to 1 and are positive
        weights = np.exp(weights)
        weights = weights / np.sum(weights)

        return weights

    def evaluate(
        self, eval_env, n_episodes: int = 10, deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate agent performance

        Args:
            eval_env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            Dictionary of performance metrics
        """
        episode_results = []

        for episode in range(n_episodes):
            eval_env.reset()
            done = False
            episode_data = {"returns": [], "weights_history": [], "trades": 0}

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)

                episode_data["returns"].append(info.get("step_return", 0))
                episode_data["weights_history"].append(info.get("weights", {}))
                episode_data["trades"] += info.get("trades_executed", 0)

            # Get episode summary
            episode_summary = info
            episode_data.update(episode_summary)
            episode_results.append(episode_data)

        # Aggregate results
        metrics = {
            "mean_return": np.mean([ep["total_return"] for ep in episode_results]),
            "std_return": np.std([ep["total_return"] for ep in episode_results]),
            "mean_sharpe": np.mean([ep["sharpe_ratio"] for ep in episode_results]),
            "mean_max_drawdown": np.mean([ep["max_drawdown"] for ep in episode_results]),
            "mean_trades": np.mean([ep["trades"] for ep in episode_results]),
            "mean_win_rate": np.mean([ep["win_rate"] for ep in episode_results]),
        }

        return metrics

    def get_portfolio_weights(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Get portfolio weights as a dictionary

        Args:
            observation: Current state

        Returns:
            Dictionary mapping symbols to weights
        """
        weights, _ = self.predict(observation, deterministic=True)

        # Map to symbol names (assuming env provides this)
        weight_dict = {}
        for i, symbol in enumerate(self.env.get_attr("symbols")[0]):
            weight_dict[symbol] = float(weights[i])
        weight_dict["cash"] = float(weights[-1])

        return weight_dict

    def save(self, path: str):
        """Save model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model"""
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")

    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress and statistics"""
        if hasattr(self.model, "logger"):
            return {
                "n_timesteps": self.model.num_timesteps,
                "learning_rate": self.model.learning_rate,
                "clip_range": (
                    self.model.clip_range(1.0)
                    if callable(self.model.clip_range)
                    else self.model.clip_range
                ),
            }
        return {}


class PortfolioAgentFactory:
    """Factory for creating portfolio agents with different configurations"""

    @staticmethod
    def create_conservative_agent(env, n_assets: int) -> PortfolioAgent:
        """Create conservative portfolio agent"""
        config = {
            "learning_rate": 1e-4,
            "clip_range": 0.1,
            "ent_coef": 0.02,  # Higher entropy for more exploration
            "n_steps": 4096,  # Longer episodes for stability
        }
        return PortfolioAgent(env, n_assets, config)

    @staticmethod
    def create_aggressive_agent(env, n_assets: int) -> PortfolioAgent:
        """Create aggressive portfolio agent"""
        config = {
            "learning_rate": 5e-4,
            "clip_range": 0.3,
            "ent_coef": 0.005,  # Lower entropy for exploitation
            "n_steps": 1024,  # Shorter episodes for faster adaptation
        }
        return PortfolioAgent(env, n_assets, config)

    @staticmethod
    def create_balanced_agent(env, n_assets: int) -> PortfolioAgent:
        """Create balanced portfolio agent"""
        config = {
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "n_steps": 2048,
        }
        return PortfolioAgent(env, n_assets, config)
