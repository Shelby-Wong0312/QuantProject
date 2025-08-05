#!/usr/bin/env python
"""
Quick Start Script for Intelligent Quantitative Trading System
快速啟動智能量化交易系統
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy',
        'pandas',
        'torch',
        'gym',
        'plotly',
        'dash'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    return True


def setup_environment():
    """Setup environment variables"""
    env_file = project_root / '.env'
    env_example = project_root / '.env.example'
    
    if not env_file.exists() and env_example.exists():
        logger.warning(".env file not found. Copying from .env.example")
        import shutil
        shutil.copy(env_example, env_file)
        logger.info("Please edit .env file with your API credentials")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    return True


def run_visualization_dashboard():
    """Run the visualization dashboard"""
    logger.info("Starting visualization dashboard...")
    
    try:
        from src.visualization.dashboard.app import TradingDashboard
        
        dashboard = TradingDashboard()
        logger.info("Dashboard is running at http://localhost:8050")
        logger.info("Press Ctrl+C to stop")
        dashboard.run(debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return False
    
    return True


def run_backtest_demo():
    """Run a simple backtest demonstration"""
    logger.info("Running backtest demonstration...")
    
    try:
        from src.backtesting.example_usage import run_example_backtest
        
        # Run backtest with sample data
        results = run_example_backtest(
            symbol='AAPL',
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            initial_capital=100000
        )
        
        logger.info(f"Backtest completed!")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return False
    
    return True


def train_simple_model():
    """Train a simple LSTM model"""
    logger.info("Training simple LSTM model...")
    
    try:
        from src.models.ml_models.example_usage import train_lstm_example
        
        # Train on sample data
        model_path = train_lstm_example(
            symbol='AAPL',
            epochs=10,
            save_path='./models/lstm_demo/'
        )
        
        logger.info(f"Model trained and saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False
    
    return True


def run_paper_trading():
    """Run paper trading simulation"""
    logger.info("Starting paper trading simulation...")
    
    try:
        from src.integration.main_controller import MainController
        
        # Configure for paper trading
        config = {
            'mode': 'paper',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'initial_capital': 100000,
            'update_interval': 60  # seconds
        }
        
        controller = MainController(config)
        
        logger.info("Paper trading started. Press Ctrl+C to stop")
        controller.run()
        
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    except Exception as e:
        logger.error(f"Paper trading failed: {e}")
        return False
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quick Start for Intelligent Quantitative Trading System"
    )
    
    parser.add_argument(
        '--mode',
        choices=['dashboard', 'backtest', 'train', 'paper', 'check'],
        default='check',
        help='Operating mode'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install missing dependencies first")
        return 1
    
    # Setup environment
    if not setup_environment():
        logger.warning("Please configure .env file before running trading modes")
    
    # Create necessary directories
    directories = ['logs', 'models', 'results', 'data']
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
    
    # Run selected mode
    if args.mode == 'check':
        logger.info("System check completed!")
        logger.info("Available modes:")
        logger.info("  --mode dashboard : Run visualization dashboard")
        logger.info("  --mode backtest  : Run backtest demonstration")
        logger.info("  --mode train     : Train a simple model")
        logger.info("  --mode paper     : Run paper trading")
        
    elif args.mode == 'dashboard':
        run_visualization_dashboard()
        
    elif args.mode == 'backtest':
        run_backtest_demo()
        
    elif args.mode == 'train':
        train_simple_model()
        
    elif args.mode == 'paper':
        run_paper_trading()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())