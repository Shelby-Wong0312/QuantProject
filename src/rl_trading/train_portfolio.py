"""
Training script for Portfolio Management Agent
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from rl_trading.environments.portfolio_env import PortfolioTradingEnvironment
from rl_trading.agents.portfolio_agent import PortfolioAgentFactory
from rl_trading.training.callbacks import TradingMetricsCallback

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_market_data(symbols: list, data_path: str = None) -> dict:
    """
    Load market data for multiple symbols

    Args:
        symbols: List of stock symbols
        data_path: Path to data directory

    Returns:
        Dictionary of symbol -> DataFrame
    """
    market_data = {}

    for symbol in symbols:
        # In production, load real data from data_path
        # For now, generate synthetic data
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")

        # Generate correlated returns
        np.random.seed(hash(symbol) % 1000)
        market_return = np.random.normal(0.0002, 0.02, len(dates))
        idiosyncratic_return = np.random.normal(0, 0.01, len(dates))

        # Combine with market factor
        returns = 0.7 * market_return + 0.3 * idiosyncratic_return
        prices = 100 * np.exp(np.cumsum(returns))

        market_data[symbol] = pd.DataFrame(
            {
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                "low": prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
                "close": prices,
                "volume": np.random.lognormal(15, 0.5, len(dates)),
            },
            index=dates,
        )

    logger.info(f"Loaded market data for {len(symbols)} symbols")
    return market_data


def create_train_test_envs(args):
    """Create training and testing environments"""

    # Load market data
    market_data = load_market_data(args.symbols)

    # Split data
    split_date = pd.Timestamp("2022-01-01")
    train_data = {s: data[data.index < split_date] for s, data in market_data.items()}
    test_data = {s: data[data.index >= split_date] for s, data in market_data.items()}

    # Create environments
    train_env = PortfolioTradingEnvironment(
        args.symbols,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        max_steps_per_episode=args.episode_length,
        rebalance_frequency=args.rebalance_freq,
    )
    train_env.set_market_data(train_data)

    test_env = PortfolioTradingEnvironment(
        args.symbols,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        max_steps_per_episode=args.episode_length,
        rebalance_frequency=args.rebalance_freq,
    )
    test_env.set_market_data(test_data)

    return train_env, test_env


def main(args):
    """Main training function"""

    # Create environments
    logger.info(f"Creating environments for symbols: {args.symbols}")
    train_env, test_env = create_train_test_envs(args)

    # Create agent based on strategy type
    logger.info(f"Creating {args.strategy} agent")

    if args.strategy == "conservative":
        agent = PortfolioAgentFactory.create_conservative_agent(
            train_env, len(args.symbols)
        )
    elif args.strategy == "aggressive":
        agent = PortfolioAgentFactory.create_aggressive_agent(
            train_env, len(args.symbols)
        )
    else:
        agent = PortfolioAgentFactory.create_balanced_agent(
            train_env, len(args.symbols)
        )

    # Override with command line parameters if provided
    if args.learning_rate:
        agent.config["learning_rate"] = args.learning_rate
    if args.batch_size:
        agent.config["batch_size"] = args.batch_size

    # Create callbacks
    callbacks = []

    # Trading metrics callback
    metrics_callback = TradingMetricsCallback(
        log_dir=Path(args.output_dir) / "logs", log_freq=1000
    )
    callbacks.append(metrics_callback)

    # Train agent
    logger.info(f"Starting training for {args.total_timesteps} timesteps")

    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=test_env,
        eval_freq=args.eval_freq,
        save_path=args.output_dir,
        callback_list=callbacks,
    )

    # Final evaluation
    logger.info("Running final evaluation on test set")

    final_metrics = agent.evaluate(test_env, n_episodes=20, deterministic=True)

    # Save results
    results = {
        "training_config": vars(args),
        "agent_config": agent.config,
        "final_metrics": final_metrics,
        "symbols": args.symbols,
        "training_completed": datetime.now().isoformat(),
    }

    results_path = Path(args.output_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PORTFOLIO TRAINING COMPLETED")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Total timesteps: {args.total_timesteps}")
    print("\nFinal Performance:")
    print(f"  Mean Return: {final_metrics['mean_return']:.2%}")
    print(f"  Sharpe Ratio: {final_metrics['mean_sharpe']:.2f}")
    print(f"  Max Drawdown: {final_metrics['mean_max_drawdown']:.2%}")
    print(f"  Win Rate: {final_metrics['mean_win_rate']:.2%}")
    print(f"  Avg Trades: {final_metrics['mean_trades']:.1f}")
    print("=" * 60)

    return agent, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train portfolio management agent")

    # Portfolio settings
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        help="Stock symbols to trade",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100000, help="Initial capital"
    )
    parser.add_argument(
        "--transaction-cost", type=float, default=0.001, help="Transaction cost rate"
    )
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate")
    parser.add_argument(
        "--episode-length", type=int, default=252, help="Trading days per episode"
    )
    parser.add_argument(
        "--rebalance-freq", type=int, default=5, help="Rebalancing frequency (days)"
    )

    # Agent settings
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["conservative", "aggressive", "balanced"],
        default="balanced",
        help="Trading strategy type",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )

    # Training settings
    parser.add_argument(
        "--total-timesteps", type=int, default=500000, help="Total training timesteps"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./models/portfolio", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    main(args)
