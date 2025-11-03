"""
Example usage of the RL trading environment
"""

import matplotlib.pyplot as plt
import logging

from environments import TradingEnvironment
from environments.reward_calculator import RewardConfig
from configs.env_config import get_config
from utils.portfolio_tracker import PortfolioTracker
from utils.risk_manager import RiskManager, RiskLimits

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_environment_basic():
    """Test basic environment functionality"""
    print("\n" + "=" * 60)
    print("TESTING BASIC ENVIRONMENT FUNCTIONALITY")
    print("=" * 60)

    # Create environment with default config
    env = TradingEnvironment(
        symbol="AAPL", initial_capital=10000, max_steps=100, action_type="discrete"
    )

    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test random actions
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\nStep {i+1}:")
        print(f"  Action: {env.action_space_handler.get_action_meanings()[action]}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
        print(f"  Position: {info['position']} shares")

        if done:
            break

    print(f"\nTotal reward: {total_reward:.4f}")

    # Get episode summary
    summary = env.get_episode_summary()
    print("\nEpisode Summary:")
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )


def test_environment_with_strategy():
    """Test environment with a simple trading strategy"""
    print("\n" + "=" * 60)
    print("TESTING WITH SIMPLE MOMENTUM STRATEGY")
    print("=" * 60)

    # Create environment
    env = TradingEnvironment(
        symbol="AAPL", initial_capital=10000, max_steps=252, action_type="discrete"
    )

    state = env.reset()
    done = False

    portfolio_values = []
    positions = []

    while not done:
        # Simple momentum strategy
        # Extract some features from state (this is just an example)
        # In practice, you'd use the actual state features

        # Get current info
        info = env._get_info()

        # Simple rule: Buy if price trending up, sell if trending down
        if info["price_trend"] > 0.001 and info["position"] == 0:
            action = 4  # BUY_100
        elif info["price_trend"] < -0.001 and info["position"] > 0:
            action = 8  # SELL_100
        else:
            action = 0  # HOLD

        state, reward, done, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        positions.append(info["position"])

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value
    ax1.plot(portfolio_values)
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_ylabel("Value ($)")
    ax1.grid(True)

    # Position
    ax2.plot(positions)
    ax2.set_title("Position Over Time")
    ax2.set_ylabel("Shares")
    ax2.set_xlabel("Step")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("momentum_strategy_results.png")
    print("\nResults saved to momentum_strategy_results.png")

    # Get episode summary
    summary = env.get_episode_summary()
    print("\nStrategy Performance:")
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )


def test_risk_management():
    """Test risk management integration"""
    print("\n" + "=" * 60)
    print("TESTING RISK MANAGEMENT")
    print("=" * 60)

    # Create risk manager
    risk_limits = RiskLimits(
        max_position_size=0.25,
        max_daily_loss=0.02,
        max_drawdown=0.05,
        max_trades_per_day=5,
    )
    risk_manager = RiskManager(risk_limits)

    # Create portfolio tracker
    portfolio_tracker = PortfolioTracker(initial_capital=10000)

    # Simulate some trades
    prices = {"AAPL": 150.0}
    portfolio_value = 10000

    trades = [
        ("BUY", 50, 150.0),
        ("BUY", 30, 151.0),
        ("SELL", 40, 152.0),
        ("SELL", 40, 149.0),
        ("BUY", 100, 148.0),
    ]

    for action, shares, price in trades:
        # Update price
        prices["AAPL"] = price

        # Check if trade allowed
        current_positions = {
            "AAPL": portfolio_tracker.positions.get("AAPL", 0).shares * price
        }
        allowed, reason = risk_manager.check_trade_allowed(
            action,
            shares,
            price,
            portfolio_value,
            current_positions,
            portfolio_tracker.cash,
        )

        print(f"\nTrade: {action} {shares} @ ${price}")
        print(f"  Allowed: {allowed}")
        print(f"  Reason: {reason}")

        if allowed:
            # Execute trade
            success = portfolio_tracker.execute_trade("AAPL", action, shares, price)
            if success:
                # Update portfolio value
                portfolio_value = portfolio_tracker.get_portfolio_value(prices)

                # Update risk metrics
                trade_pnl = 0  # Would calculate actual P&L
                risk_manager.update_metrics(trade_pnl, portfolio_value, portfolio_value)

    # Get final metrics
    print("\nPortfolio Summary:")
    position_summary = portfolio_tracker.get_position_summary(prices)
    print(position_summary)

    print("\nRisk Metrics:")
    risk_metrics = risk_manager.get_risk_metrics(portfolio_value)
    for key, value in risk_metrics.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )


def test_different_configs():
    """Test environment with different configurations"""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 60)

    configs = {
        "Conservative": get_config("conservative"),
        "Aggressive": get_config("aggressive"),
        "Day Trading": get_config("day_trading"),
    }

    for name, config in configs.items():
        print(f"\n--- {name} Configuration ---")

        # Create environment
        env = TradingEnvironment(
            symbol="AAPL",
            initial_capital=config.initial_capital,
            max_steps=min(50, config.max_steps_per_episode),  # Limit for demo
            action_type=config.action_type,
            reward_config=RewardConfig(
                volatility_penalty=config.risk_penalties["volatility"],
                max_drawdown_penalty=config.risk_penalties["drawdown"],
                exposure_penalty=config.risk_penalties["exposure"],
            ),
        )

        # Run episode with random actions
        state = env.reset()
        total_reward = 0
        n_trades = 0

        for _ in range(50):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            total_reward += reward

            if info["trade_details"]["shares"] != 0:
                n_trades += 1

            if done:
                break

        summary = env.get_episode_summary()
        print(f"  Total Return: {summary['total_return']:.2%}")
        print(f"  Total Trades: {n_trades}")
        print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {summary.get('max_drawdown', 0):.2%}")


def demonstrate_state_features():
    """Demonstrate the state features being used"""
    print("\n" + "=" * 60)
    print("STATE FEATURE DEMONSTRATION")
    print("=" * 60)

    # Create environment
    env = TradingEnvironment(symbol="AAPL", initial_capital=10000, max_steps=10)

    state = env.reset()

    # Get feature names
    feature_names = env.state_processor.get_feature_names()

    print(f"Total features: {len(feature_names)}")
    print("\nFeature categories:")

    # Group features by category
    categories = {
        "Market": [f for f in feature_names if f.startswith("market_")],
        "Technical": [f for f in feature_names if f.startswith("tech_")],
        "LSTM": [f for f in feature_names if f.startswith("lstm_")],
        "Sentiment": [f for f in feature_names if f.startswith("sentiment_")],
        "Portfolio": [f for f in feature_names if f.startswith("portfolio_")],
        "Time": [f for f in feature_names if f.startswith("time_")],
    }

    for category, features in categories.items():
        print(f"\n{category} features ({len(features)}):")
        for feat in features[:5]:  # Show first 5
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")

    # Show sample state values
    print("\nSample state values:")
    for i in range(min(10, len(state))):
        print(f"  {feature_names[i]}: {state[i]:.4f}")


if __name__ == "__main__":
    # Run all tests
    test_environment_basic()
    test_environment_with_strategy()
    test_risk_management()
    test_different_configs()
    demonstrate_state_features()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
