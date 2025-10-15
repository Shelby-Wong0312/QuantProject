"""
Neural network architectures for RL agents
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Type
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading state representation
    Handles the multi-dimensional trading features
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        use_lstm: bool = False,
        lstm_hidden_size: int = 128,
    ):
        """
        Initialize feature extractor

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            use_lstm: Whether to use LSTM for temporal features
            lstm_hidden_size: LSTM hidden size
        """
        super().__init__(observation_space, features_dim)

        self.use_lstm = use_lstm
        input_dim = observation_space.shape[0]

        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

        # Optional LSTM for temporal dependencies
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=features_dim,
                hidden_size=lstm_hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.lstm_projection = nn.Linear(lstm_hidden_size, features_dim)

        logger.info(f"Initialized TradingFeatureExtractor with output dim {features_dim}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: Input observations

        Returns:
            Extracted features
        """
        # Extract features
        features = self.feature_net(observations)

        # Apply LSTM if enabled
        if self.use_lstm and len(observations.shape) == 3:
            # Reshape for LSTM if needed
            lstm_out, _ = self.lstm(features)
            features = self.lstm_projection(lstm_out[:, -1, :])  # Take last output

        return features


class TradingActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for trading
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        actor_hidden_sizes: List[int] = [256, 128],
        critic_hidden_sizes: List[int] = [256, 128],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        """
        Initialize Actor-Critic network

        Args:
            feature_dim: Input feature dimension
            action_dim: Action space dimension
            actor_hidden_sizes: Hidden layer sizes for actor
            critic_hidden_sizes: Hidden layer sizes for critic
            activation_fn: Activation function
        """
        super().__init__()

        # Build actor network
        actor_layers = []
        last_dim = feature_dim

        for hidden_size in actor_hidden_sizes:
            actor_layers.extend(
                [nn.Linear(last_dim, hidden_size), activation_fn(), nn.Dropout(0.1)]
            )
            last_dim = hidden_size

        self.actor_net = nn.Sequential(*actor_layers)
        self.action_net = nn.Linear(last_dim, action_dim)

        # Build critic network
        critic_layers = []
        last_dim = feature_dim

        for hidden_size in critic_hidden_sizes:
            critic_layers.extend(
                [nn.Linear(last_dim, hidden_size), activation_fn(), nn.Dropout(0.1)]
            )
            last_dim = hidden_size

        self.critic_net = nn.Sequential(*critic_layers)
        self.value_net = nn.Linear(last_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        # Actor network
        for module in self.actor_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Action head
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.constant_(self.action_net.bias, 0.0)

        # Critic network
        for module in self.critic_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Value head
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)
        nn.init.constant_(self.value_net.bias, 0.0)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for actor"""
        return self.action_net(self.actor_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for critic"""
        return self.value_net(self.critic_net(features))


class TradingPolicy(ActorCriticPolicy):
    """
    Custom policy for trading environment
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        use_sde: bool = False,
        use_lstm: bool = False,
        **kwargs,
    ):
        """
        Initialize trading policy

        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            use_sde: Use state dependent exploration
            use_lstm: Use LSTM in feature extractor
        """
        # Set custom feature extractor
        kwargs["features_extractor_class"] = TradingFeatureExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256, "use_lstm": use_lstm}

        super().__init__(observation_space, action_space, lr_schedule, use_sde=use_sde, **kwargs)

        logger.info("Initialized TradingPolicy")


class AttentionLayer(nn.Module):
    """
    Attention mechanism for feature importance
    """

    def __init__(self, feature_dim: int):
        """
        Initialize attention layer

        Args:
            feature_dim: Feature dimension
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(), nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism

        Args:
            features: Input features

        Returns:
            Weighted features and attention weights
        """
        # Calculate attention scores
        attention_scores = self.attention(features)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention
        weighted_features = features * attention_weights

        return weighted_features, attention_weights


class EnsembleNetwork(nn.Module):
    """
    Ensemble of multiple networks for robustness
    """

    def __init__(self, feature_dim: int, action_dim: int, n_models: int = 3):
        """
        Initialize ensemble network

        Args:
            feature_dim: Input feature dimension
            action_dim: Action dimension
            n_models: Number of models in ensemble
        """
        super().__init__()

        self.n_models = n_models

        # Create multiple actor-critic networks
        self.models = nn.ModuleList(
            [TradingActorCriticNetwork(feature_dim, action_dim) for _ in range(n_models)]
        )

    def forward(self, features: torch.Tensor, return_std: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble

        Args:
            features: Input features
            return_std: Whether to return standard deviation

        Returns:
            Dictionary with ensemble predictions
        """
        # Get predictions from all models
        action_preds = []
        value_preds = []

        for model in self.models:
            action_preds.append(model.forward_actor(features))
            value_preds.append(model.forward_critic(features))

        # Stack predictions
        action_preds = torch.stack(action_preds, dim=0)
        value_preds = torch.stack(value_preds, dim=0)

        # Calculate mean and std
        action_mean = action_preds.mean(dim=0)
        value_mean = value_preds.mean(dim=0)

        results = {"action": action_mean, "value": value_mean}

        if return_std:
            results["action_std"] = action_preds.std(dim=0)
            results["value_std"] = value_preds.std(dim=0)

        return results
