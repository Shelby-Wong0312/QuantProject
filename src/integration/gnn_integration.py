"""
GNN Integration Module
整合GNN關聯分析到投資組合環境
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from sensory_models.relation_analyzer import StockRelationAnalyzer, RelationFeatureExtractor
from rl_trading.environments.portfolio_env import PortfolioTradingEnvironment

logger = logging.getLogger(__name__)


class GNNEnhancedPortfolioEnv(PortfolioTradingEnvironment):
    """
    GNN增強的投資組合環境
    整合股票關聯分析到狀態空間
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
        use_gnn_features: bool = True,
        gnn_update_frequency: int = 20,
        **kwargs,
    ):
        """
        初始化GNN增強環境

        Args:
            symbols: 股票代碼列表
            initial_capital: 初始資金
            use_gnn_features: 是否使用GNN特徵
            gnn_update_frequency: GNN更新頻率（天）
            **kwargs: 其他環境參數
        """
        super().__init__(symbols, initial_capital, **kwargs)

        self.use_gnn_features = use_gnn_features
        self.gnn_update_frequency = gnn_update_frequency

        if self.use_gnn_features:
            # Initialize GNN components
            self.relation_analyzer = StockRelationAnalyzer(
                model_config={"hidden_dim": 128, "num_layers": 3, "dropout": 0.1},
                device="cuda" if self._check_cuda_available() else "cpu",
            )

            self.feature_extractor = RelationFeatureExtractor(self.relation_analyzer)

            # Update observation space dimension
            self.gnn_feature_dim = len(symbols) * 5  # 5 GNN features per symbol
            self.state_dim = self.observation_space.shape[0] + self.gnn_feature_dim

            # Update observation space
            import gymnasium.spaces as spaces

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
            )

            # GNN update counter
            self.steps_since_gnn_update = 0
            self.cached_gnn_features = None

            logger.info(
                f"Initialized GNN-enhanced environment with {self.gnn_feature_dim} additional features"
            )

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def reset(self) -> np.ndarray:
        """Reset environment with GNN features"""
        # Reset base environment
        base_obs = super().reset()

        if not self.use_gnn_features:
            return base_obs

        # Update GNN features
        self._update_gnn_features()

        # Combine features
        gnn_features = self._get_gnn_features()
        enhanced_obs = np.concatenate([base_obs, gnn_features])

        return enhanced_obs

    def step(self, action: np.ndarray):
        """Step with GNN feature updates"""
        # Execute base step
        base_obs, reward, done, info = super().step(action)

        if not self.use_gnn_features:
            return base_obs, reward, done, info

        # Update GNN features if needed
        self.steps_since_gnn_update += 1
        if self.steps_since_gnn_update >= self.gnn_update_frequency:
            self._update_gnn_features()
            self.steps_since_gnn_update = 0

        # Enhance observation with GNN features
        gnn_features = self._get_gnn_features()
        enhanced_obs = np.concatenate([base_obs, gnn_features])

        # Add GNN info
        info["gnn_updated"] = self.steps_since_gnn_update == 0
        info["gnn_features"] = gnn_features

        return enhanced_obs, reward, done, info

    def _update_gnn_features(self):
        """Update GNN analysis"""
        try:
            # Prepare historical data for GNN
            lookback = 60  # 60 days of history
            end_idx = self.current_step
            start_idx = max(0, end_idx - lookback)

            price_data = {}
            for symbol in self.symbols:
                if symbol in self.market_data and len(self.market_data[symbol]) > start_idx:
                    price_data[symbol] = self.market_data[symbol].iloc[start_idx:end_idx]

            if len(price_data) == len(self.symbols):
                # Analyze relations
                results = self.relation_analyzer.analyze_relations(price_data, save_results=False)

                # Cache features
                current_prices = {
                    symbol: self.market_data[symbol].iloc[self.current_step]["close"]
                    for symbol in self.symbols
                }

                self.cached_gnn_features = self.feature_extractor.extract_relation_features(
                    self.symbols, current_prices
                )
            else:
                logger.warning("Insufficient data for GNN update")
                self.cached_gnn_features = np.zeros(self.gnn_feature_dim)

        except Exception as e:
            logger.error(f"Error updating GNN features: {e}")
            self.cached_gnn_features = np.zeros(self.gnn_feature_dim)

    def _get_gnn_features(self) -> np.ndarray:
        """Get current GNN features"""
        if self.cached_gnn_features is None:
            return np.zeros(self.gnn_feature_dim)
        return self.cached_gnn_features

    def _calculate_reward(self, portfolio_return: float) -> float:
        """Enhanced reward calculation with correlation consideration"""
        # Base reward
        base_reward = super()._calculate_reward(portfolio_return)

        if not self.use_gnn_features or self.cached_gnn_features is None:
            return base_reward

        # Add correlation-based penalty/reward
        try:
            # Get correlation matrix from analyzer
            if "latest" in self.relation_analyzer.analysis_cache:
                corr_matrix = self.relation_analyzer.analysis_cache["latest"]["correlation_matrix"]

                # Calculate portfolio correlation
                weights = np.array([self.portfolio_state.weights.get(s, 0) for s in self.symbols])
                weights = weights[:-1]  # Exclude cash

                # Portfolio correlation (weighted average of pairwise correlations)
                portfolio_corr = 0
                weight_sum = 0

                for i in range(len(self.symbols)):
                    for j in range(i + 1, len(self.symbols)):
                        if weights[i] > 0 and weights[j] > 0:
                            portfolio_corr += weights[i] * weights[j] * corr_matrix[i, j]
                            weight_sum += weights[i] * weights[j]

                if weight_sum > 0:
                    avg_correlation = portfolio_corr / weight_sum

                    # Penalize high correlation (encourage diversification)
                    correlation_penalty = -0.1 * max(0, avg_correlation - 0.5)
                    base_reward += correlation_penalty

        except Exception as e:
            logger.error(f"Error calculating correlation reward: {e}")

        return base_reward


class GNNPortfolioAgent:
    """
    Portfolio agent that leverages GNN features
    """

    def __init__(
        self,
        env: GNNEnhancedPortfolioEnv,
        base_agent_class: Any,
        agent_config: Optional[Dict] = None,
    ):
        """
        Initialize GNN-aware portfolio agent

        Args:
            env: GNN-enhanced environment
            base_agent_class: Base agent class to enhance
            agent_config: Agent configuration
        """
        self.env = env
        self.agent_config = agent_config or {}

        # Create base agent with modified feature extractor
        self.base_agent = base_agent_class(
            env=env, n_assets=len(env.symbols), config=self.agent_config
        )

        logger.info("Initialized GNN-enhanced portfolio agent")

    def train(self, **kwargs):
        """Train the agent"""
        return self.base_agent.train(**kwargs)

    def predict(self, observation, **kwargs):
        """Make predictions"""
        return self.base_agent.predict(observation, **kwargs)

    def evaluate(self, **kwargs):
        """Evaluate agent performance"""
        return self.base_agent.evaluate(**kwargs)

    def save(self, path: str):
        """Save agent and GNN model"""
        # Save base agent
        self.base_agent.save(path)

        # Save GNN model
        gnn_path = Path(path).parent / "gnn_model.pth"
        self.env.relation_analyzer.save_model(str(gnn_path))

        logger.info(f"Saved GNN-enhanced agent to {path}")

    def load(self, path: str):
        """Load agent and GNN model"""
        # Load base agent
        self.base_agent.load(path)

        # Load GNN model
        gnn_path = Path(path).parent / "gnn_model.pth"
        if gnn_path.exists():
            self.env.relation_analyzer.load_model(str(gnn_path))
            logger.info("Loaded GNN model")


def create_gnn_enhanced_environment(
    symbols: List[str], market_data: Dict[str, pd.DataFrame], **env_kwargs
) -> GNNEnhancedPortfolioEnv:
    """
    Factory function to create GNN-enhanced environment

    Args:
        symbols: List of stock symbols
        market_data: Market data dictionary
        **env_kwargs: Additional environment parameters

    Returns:
        GNN-enhanced portfolio environment
    """
    # Create environment
    env = GNNEnhancedPortfolioEnv(symbols=symbols, use_gnn_features=True, **env_kwargs)

    # Set market data
    env.set_market_data(market_data)

    # Pre-compute initial GNN features
    env._update_gnn_features()

    return env


# Example usage
if __name__ == "__main__":
    # Test GNN integration
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Generate sample data
    market_data = {}
    for symbol in symbols:
        dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, len(dates))))

        market_data[symbol] = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.lognormal(15, 0.5, len(dates)),
            },
            index=dates,
        )

    # Create GNN-enhanced environment
    env = create_gnn_enhanced_environment(
        symbols, market_data=market_data, initial_capital=100000, transaction_cost=0.001
    )

    # Test environment
    env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Base features: {env.observation_space.shape[0] - env.gnn_feature_dim}")
    print(f"GNN features: {env.gnn_feature_dim}")

    # Test step
    action = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Equal weights
    obs, reward, done, info = env.step(action)

    print("\nStep completed:")
    print(f"Reward: {reward:.4f}")
    print(f"GNN updated: {info.get('gnn_updated', False)}")
