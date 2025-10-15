"""
Unit tests for Portfolio Trading Environment
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl_trading.environments.portfolio_env import PortfolioTradingEnvironment, PortfolioState
from src.rl_trading.agents.portfolio_agent import PortfolioAgent, PortfolioAgentFactory


class TestPortfolioEnvironment(unittest.TestCase):
    """Test cases for portfolio trading environment"""

    def setUp(self):
        """Set up test environment"""
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        self.env = PortfolioTradingEnvironment(
            symbols=self.symbols,
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005,
            max_steps_per_episode=50,
            lookback_window=20,
            rebalance_frequency=5,
        )

    def test_environment_initialization(self):
        """Test environment initialization"""
        self.assertEqual(len(self.symbols), self.env.n_assets)
        self.assertEqual(self.env.initial_capital, 100000)

        # Check observation space
        expected_obs_dim = len(self.symbols) * self.env.features_per_asset + 10
        self.assertEqual(self.env.observation_space.shape[0], expected_obs_dim)

        # Check action space (n_assets + 1 for cash)
        self.assertEqual(self.env.action_space.shape[0], len(self.symbols) + 1)

    def test_reset(self):
        """Test environment reset"""
        obs = self.env.reset()

        # Check observation shape
        self.assertEqual(obs.shape[0], self.env.state_dim)

        # Check portfolio state
        self.assertEqual(self.env.portfolio_state.cash, self.env.initial_capital)
        self.assertEqual(self.env.portfolio_state.total_value, self.env.initial_capital)

        # Check all positions are zero
        for symbol in self.symbols:
            self.assertEqual(self.env.portfolio_state.positions[symbol], 0)

        # Check weights (should be all cash)
        self.assertEqual(self.env.portfolio_state.weights["cash"], 1.0)

    def test_action_validation(self):
        """Test action validation and normalization"""
        # Test unnormalized action
        action = np.array([0.2, 0.3, 0.1, 0.2, 0.1, 0.5])  # Sum > 1
        validated = self.env._validate_action(action)

        # Should sum to 1
        self.assertAlmostEqual(np.sum(validated), 1.0, places=6)

        # Test all zeros
        action_zeros = np.zeros(6)
        validated_zeros = self.env._validate_action(action_zeros)
        self.assertEqual(validated_zeros[-1], 1.0)  # Should be all cash

        # Test negative values
        action_neg = np.array([-0.1, 0.3, 0.2, 0.3, 0.2, 0.3])
        validated_neg = self.env._validate_action(action_neg)
        self.assertTrue(np.all(validated_neg >= 0))
        self.assertAlmostEqual(np.sum(validated_neg), 1.0, places=6)

    def test_step_without_rebalance(self):
        """Test step when rebalancing is not allowed"""
        obs = self.env.reset()

        # First step (rebalance allowed)
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Allocate to stocks
        obs1, reward1, done1, info1 = self.env.step(action)

        self.assertTrue(info1["can_rebalance"])
        self.assertGreater(info1["trades_executed"], 0)

        # Second step (no rebalance)
        obs2, reward2, done2, info2 = self.env.step(action)

        self.assertFalse(info2["can_rebalance"])
        self.assertEqual(info2["trades_executed"], 0)

    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing logic"""
        self.env.reset()

        # Set up initial prices
        prices = {symbol: 100.0 for symbol in self.symbols}

        # Test rebalancing
        target_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # 90% stocks, 10% cash
        trades = self.env._rebalance_portfolio(target_weights, prices)

        # Check trades were executed
        self.assertGreater(len(trades), 0)

        # Check cash decreased
        self.assertLess(self.env.portfolio_state.cash, self.env.initial_capital)

        # Check positions were created
        total_positions = sum(self.env.portfolio_state.positions.values())
        self.assertGreater(total_positions, 0)

    def test_reward_calculation(self):
        """Test reward calculation"""
        self.env.reset()

        # Test positive return
        positive_return = 0.02
        reward_pos = self.env._calculate_reward(positive_return)
        self.assertGreater(reward_pos, 0)

        # Test negative return
        negative_return = -0.02
        reward_neg = self.env._calculate_reward(negative_return)
        self.assertLess(reward_neg, 0)

        # Test with diversification
        self.env.portfolio_state.weights = {
            "AAPL": 0.2,
            "GOOGL": 0.2,
            "MSFT": 0.2,
            "AMZN": 0.2,
            "TSLA": 0.1,
            "cash": 0.1,
        }
        reward_div = self.env._calculate_reward(0.0)
        self.assertGreater(reward_div, 0)  # Should get diversification bonus

    def test_episode_completion(self):
        """Test full episode execution"""
        obs = self.env.reset()

        episode_rewards = []
        done = False
        steps = 0

        while not done and steps < 100:  # Safety limit
            # Random action
            action = np.random.dirichlet(np.ones(self.env.n_assets + 1))
            obs, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)
            steps += 1

        self.assertTrue(done)
        self.assertEqual(steps, self.env.max_steps_per_episode)

        # Check episode summary
        self.assertIn("total_return", info)
        self.assertIn("sharpe_ratio", info)
        self.assertIn("max_drawdown", info)
        self.assertIn("final_weights", info)

    def test_state_features(self):
        """Test state feature extraction"""
        self.env.reset()

        # Test asset features
        features = self.env._get_asset_features("AAPL")
        self.assertEqual(len(features), self.env.features_per_asset)

        # Test portfolio features
        portfolio_features = self.env._get_portfolio_features()
        self.assertEqual(len(portfolio_features), 10)

        # Test full observation
        obs = self.env._get_observation()
        expected_dim = self.env.n_assets * self.env.features_per_asset + 10
        self.assertEqual(len(obs), expected_dim)

    def test_performance_metrics(self):
        """Test performance metric calculations"""
        # Test Sharpe ratio
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        sharpe = self.env._calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

        # Test max drawdown
        self.env.portfolio_history = [
            {"total_value": 100000},
            {"total_value": 110000},
            {"total_value": 105000},
            {"total_value": 95000},
            {"total_value": 100000},
        ]
        max_dd = self.env._calculate_max_drawdown()
        expected_dd = (110000 - 95000) / 110000
        self.assertAlmostEqual(max_dd, expected_dd, places=4)


class TestPortfolioAgent(unittest.TestCase):
    """Test cases for portfolio agent"""

    def setUp(self):
        """Set up test environment"""
        self.symbols = ["AAPL", "GOOGL", "MSFT"]
        self.env = PortfolioTradingEnvironment(
            symbols=self.symbols, initial_capital=100000, max_steps_per_episode=20
        )

        self.agent = PortfolioAgent(
            self.env,
            n_assets=len(self.symbols),
            config={"n_steps": 128, "batch_size": 32},  # Small for testing
        )

    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.n_assets, len(self.symbols))
        self.assertIsNotNone(self.agent.model)
        self.assertEqual(self.agent.config["n_steps"], 128)

    def test_weight_normalization(self):
        """Test portfolio weight normalization"""
        # Test with raw outputs
        raw_weights = np.array([1.5, -0.5, 2.0, 0.5])
        normalized = self.agent._normalize_weights(raw_weights)

        # Check properties
        self.assertAlmostEqual(np.sum(normalized), 1.0, places=6)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))

    def test_prediction(self):
        """Test agent prediction"""
        obs = self.env.reset()

        # Get prediction
        action, states = self.agent.predict(obs, deterministic=True)

        # Check action properties
        self.assertEqual(len(action), len(self.symbols) + 1)  # +1 for cash
        self.assertAlmostEqual(np.sum(action), 1.0, places=6)
        self.assertTrue(np.all(action >= 0))

    def test_agent_factory(self):
        """Test agent factory methods"""
        # Conservative agent
        conservative = PortfolioAgentFactory.create_conservative_agent(self.env, len(self.symbols))
        self.assertEqual(conservative.config["learning_rate"], 1e-4)
        self.assertEqual(conservative.config["clip_range"], 0.1)

        # Aggressive agent
        aggressive = PortfolioAgentFactory.create_aggressive_agent(self.env, len(self.symbols))
        self.assertEqual(aggressive.config["learning_rate"], 5e-4)
        self.assertEqual(aggressive.config["clip_range"], 0.3)

        # Balanced agent
        balanced = PortfolioAgentFactory.create_balanced_agent(self.env, len(self.symbols))
        self.assertEqual(balanced.config["learning_rate"], 3e-4)
        self.assertEqual(balanced.config["clip_range"], 0.2)


class TestIntegration(unittest.TestCase):
    """Integration tests for portfolio environment and agent"""

    def test_training_loop(self):
        """Test basic training loop"""
        # Create small environment
        symbols = ["AAPL", "GOOGL"]
        env = PortfolioTradingEnvironment(symbols=symbols, max_steps_per_episode=10)

        # Create agent
        agent = PortfolioAgent(
            env,
            n_assets=len(symbols),
            config={"n_steps": 20, "batch_size": 10, "learning_rate": 1e-3},
        )

        # Train for a few steps
        initial_timesteps = agent.model.num_timesteps
        agent.model.learn(total_timesteps=100)
        final_timesteps = agent.model.num_timesteps

        # Check training occurred
        self.assertGreater(final_timesteps, initial_timesteps)

    def test_evaluation(self):
        """Test agent evaluation"""
        # Create environment
        symbols = ["AAPL", "GOOGL", "MSFT"]
        env = PortfolioTradingEnvironment(symbols=symbols, max_steps_per_episode=20)

        # Create agent
        agent = PortfolioAgent(env, n_assets=len(symbols))

        # Evaluate
        metrics = agent.evaluate(env, n_episodes=2)

        # Check metrics
        self.assertIn("mean_return", metrics)
        self.assertIn("mean_sharpe", metrics)
        self.assertIn("mean_max_drawdown", metrics)
        self.assertIn("mean_trades", metrics)

        # Check values are reasonable
        self.assertIsInstance(metrics["mean_return"], (int, float))
        self.assertIsInstance(metrics["mean_sharpe"], (int, float))
        self.assertGreaterEqual(metrics["mean_max_drawdown"], 0)
        self.assertLessEqual(metrics["mean_max_drawdown"], 1)


def run_portfolio_tests():
    """Run all portfolio tests"""
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [TestPortfolioEnvironment, TestPortfolioAgent, TestIntegration]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    unittest.main()
