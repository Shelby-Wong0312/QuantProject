"""
System runner - Entry point for the trading system
"""

import asyncio
import argparse
import logging
from pathlib import Path
import json
import signal
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from integration.main_controller import MainController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemRunner:
    """Runner for the trading system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.controller = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def run(self):
        """Run the trading system"""
        try:
            # Create controller
            self.controller = MainController(self.config_path)
            
            # Initialize components
            logger.info("Initializing system components...")
            await self.controller.initialize_components()
            
            # Start system
            logger.info("Starting trading system...")
            system_task = asyncio.create_task(self.controller.start())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Stop system
            logger.info("Stopping trading system...")
            await self.controller.stop()
            
            # Cancel system task
            system_task.cancel()
            try:
                await system_task
            except asyncio.CancelledError:
                pass
            
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"System error: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def create_default_config():
        """Create default configuration file"""
        default_config = {
            "mode": "backtest",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "initial_capital": 100000,
            "risk_limit": 0.02,
            "data_lookback_days": 30,
            "prediction_horizons": [1, 5, 20],
            "sentiment_update_interval": 3600,
            "health_check_interval": 60,
            "model_paths": {
                "lstm": "./models/lstm_predictor.h5",
                "rl_agent": "./models/ppo_agent"
            },
            "backtest_settings": {
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "commission": 0.001,
                "slippage": 0.0005
            },
            "paper_trading_settings": {
                "update_interval": 5,
                "max_positions": 5,
                "stop_loss": 0.05,
                "take_profit": 0.10
            },
            "live_trading_settings": {
                "api_key": "YOUR_API_KEY",
                "api_secret": "YOUR_API_SECRET",
                "max_daily_trades": 10,
                "max_position_size": 0.1
            }
        }
        
        config_path = Path("config/system_config.json")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration at {config_path}")
        return str(config_path)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run the intelligent quantitative trading system')
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'paper', 'live'],
        default='backtest',
        help='Trading mode'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create default config if requested
    if args.create_config:
        SystemRunner.create_default_config()
        return
    
    # Get config path
    config_path = args.config
    if not config_path:
        default_path = Path("config/system_config.json")
        if default_path.exists():
            config_path = str(default_path)
        else:
            logger.info("No configuration file found, creating default...")
            config_path = SystemRunner.create_default_config()
    
    # Override mode if specified
    if args.mode and config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['mode'] = args.mode
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create directories
    for dir_name in ['logs', 'reports', 'models', 'data', 'alerts']:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Run system
    runner = SystemRunner(config_path)
    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System failed: {str(e)}", exc_info=True)
        sys.exit(1)