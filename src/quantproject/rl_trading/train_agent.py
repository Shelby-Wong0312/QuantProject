"""
Main training script for RL trading agent
"""

import argparse
import logging
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rl_trading.training import Trainer
from rl_trading.configs.env_config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training function"""
    
    # Load configurations
    logger.info(f"Starting RL training with config: {args.env_config}")
    
    # Environment config
    env_config = get_config(args.env_config)
    env_config.symbols = [args.symbol]
    env_config.initial_capital = args.initial_capital
    env_config.max_steps_per_episode = args.episode_length
    
    # Agent config
    agent_config = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'n_envs': args.n_envs,
        'ent_coef': args.ent_coef,
        'clip_range': args.clip_range
    }
    
    # Create trainer
    trainer = Trainer(
        agent_type='PPO',
        env_config=env_config,
        agent_config=agent_config,
        experiment_name=args.experiment_name,
        results_dir=args.results_dir,
        seed=args.seed
    )
    
    # Setup environments
    logger.info("Setting up environments...")
    trainer.setup_environments()
    
    # Setup agent
    logger.info("Setting up PPO agent...")
    trainer.setup_agent()
    
    # Train
    logger.info(f"Starting training for {args.total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        n_eval_episodes=args.n_eval_episodes,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test environment...")
    test_results = trainer.evaluate(
        env_type='test' if trainer.test_env else 'val',
        n_episodes=20,
        save_results=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Experiment: {trainer.experiment_name}")
    print(f"Total training time: {trainer.training_history['training_time']:.2f} seconds")
    print("\nTest Results:")
    print(f"  Mean Return: {test_results['mean_return']:.2%}")
    print(f"  Sharpe Ratio: {test_results['mean_sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {test_results['mean_max_drawdown']:.2%}")
    print(f"  Win Rate: {test_results['mean_win_rate']:.2%}")
    print(f"  Trades per Episode: {test_results['mean_trades_per_episode']:.1f}")
    print("="*60)
    
    # Save final results summary
    summary_path = Path(args.results_dir) / trainer.experiment_name / 'training_summary.json'
    summary = {
        'experiment_name': trainer.experiment_name,
        'config': {
            'env': env_config.__dict__,
            'agent': agent_config
        },
        'training_time': trainer.training_history['training_time'],
        'test_results': test_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL trading agent')
    
    # Environment settings
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock symbol to trade')
    parser.add_argument('--env-config', type=str, default='day_trading',
                        choices=['conservative', 'aggressive', 'day_trading', 'swing_trading'],
                        help='Environment configuration preset')
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial trading capital')
    parser.add_argument('--episode-length', type=int, default=252,
                        help='Maximum steps per episode')
    
    # Agent hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')
    
    # Training settings
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Model save frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Run training
    main(args)