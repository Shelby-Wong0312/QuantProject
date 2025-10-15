"""
Main Controller - Central orchestrator for the intelligent quantitative trading system
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Import system components
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data_feed.capital_com_client import CapitalComClient
from sensory_models.lstm_model import LSTMPredictor
from sensory_models.finbert_sentiment import FinBERTAnalyzer
from rl_trading.environments.trading_env import TradingEnvironment
from rl_trading.agents.ppo_agent import PPOAgent
from backtesting_engine.event_driven_engine import EventDrivenBacktester
from integration.data_pipeline import DataPipeline
from integration.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class MainController:
    """
    Central controller that orchestrates all components of the trading system
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the main controller

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Component instances
        self.data_client = None
        self.lstm_predictor = None
        self.sentiment_analyzer = None
        self.rl_agent = None
        self.backtester = None
        self.data_pipeline = None
        self.health_monitor = None

        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.message_queue = queue.Queue()

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_pnl": 0,
            "start_time": None,
            "errors": [],
        }

        logger.info("Main Controller initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "mode": "backtest",  # 'backtest', 'paper', 'live'
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "initial_capital": 100000,
            "risk_limit": 0.02,  # 2% per trade
            "data_lookback_days": 30,
            "prediction_horizons": [1, 5, 20],
            "sentiment_update_interval": 3600,  # 1 hour
            "health_check_interval": 60,  # 1 minute
            "model_paths": {"lstm": "./models/lstm_predictor.h5", "rl_agent": "./models/ppo_agent"},
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")

        try:
            # 1. Initialize data client
            self.data_client = CapitalComClient()
            await self.data_client.connect()

            # 2. Initialize data pipeline
            self.data_pipeline = DataPipeline(
                symbols=self.config["symbols"], data_client=self.data_client
            )

            # 3. Initialize LSTM predictor
            self.lstm_predictor = LSTMPredictor(sequence_length=60, n_features=10, lstm_units=128)

            # Load pre-trained model if exists
            lstm_path = Path(self.config["model_paths"]["lstm"])
            if lstm_path.exists():
                self.lstm_predictor.load_model(str(lstm_path))
                logger.info("Loaded pre-trained LSTM model")

            # 4. Initialize sentiment analyzer
            self.sentiment_analyzer = FinBERTAnalyzer()

            # 5. Initialize RL environment and agent
            self.rl_env = TradingEnvironment(
                symbol=self.config["symbols"][0],
                initial_capital=self.config["initial_capital"],
                max_steps_per_episode=252,
            )

            self.rl_agent = PPOAgent(self.rl_env)

            # Load pre-trained agent if exists
            agent_path = Path(self.config["model_paths"]["rl_agent"])
            if agent_path.exists():
                self.rl_agent.load(str(agent_path))
                logger.info("Loaded pre-trained RL agent")

            # 6. Initialize backtesting engine
            if self.config["mode"] == "backtest":
                self.backtester = EventDrivenBacktester(
                    initial_capital=self.config["initial_capital"]
                )

            # 7. Initialize health monitor
            self.health_monitor = HealthMonitor(
                components={
                    "data_client": self.data_client,
                    "lstm_predictor": self.lstm_predictor,
                    "sentiment_analyzer": self.sentiment_analyzer,
                    "rl_agent": self.rl_agent,
                    "data_pipeline": self.data_pipeline,
                },
                check_interval=self.config["health_check_interval"],
            )

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def start(self):
        """Start the trading system"""
        logger.info(f"Starting trading system in {self.config['mode']} mode...")

        self.is_running = True
        self.performance_metrics["start_time"] = datetime.now()

        # Start health monitoring
        asyncio.create_task(self.health_monitor.start())

        # Start data pipeline
        asyncio.create_task(self.data_pipeline.start())

        # Start main trading loop
        if self.config["mode"] == "backtest":
            await self._run_backtest()
        else:
            await self._run_live_trading()

    async def _run_backtest(self):
        """Run backtesting mode"""
        logger.info("Running backtest mode...")

        try:
            # Get historical data
            start_date = datetime.now() - timedelta(days=self.config["data_lookback_days"])
            end_date = datetime.now()

            for symbol in self.config["symbols"]:
                logger.info(f"Backtesting {symbol}...")

                # Fetch historical data
                historical_data = await self.data_pipeline.get_historical_data(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )

                if historical_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue

                # Process each trading day
                for date in historical_data.index:
                    if not self.is_running:
                        break

                    await self._process_trading_signal(
                        symbol=symbol, timestamp=date, data=historical_data.loc[:date]
                    )

                # Generate backtest report
                self._generate_backtest_report(symbol)

        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            self.performance_metrics["errors"].append(
                {"timestamp": datetime.now(), "error": str(e), "mode": "backtest"}
            )

    async def _run_live_trading(self):
        """Run live/paper trading mode"""
        logger.info(f"Running {self.config['mode']} trading mode...")

        # Subscribe to real-time data
        for symbol in self.config["symbols"]:
            await self.data_pipeline.subscribe_realtime(
                symbol=symbol, callback=self._handle_realtime_data
            )

        # Main trading loop
        while self.is_running:
            try:
                # Check for new messages
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    await self._process_trading_signal(**message)

                # Brief sleep to prevent CPU spinning
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Live trading error: {str(e)}")
                self.performance_metrics["errors"].append(
                    {"timestamp": datetime.now(), "error": str(e), "mode": self.config["mode"]}
                )

    async def _handle_realtime_data(self, symbol: str, data: Dict[str, Any]):
        """Handle real-time data updates"""
        self.message_queue.put({"symbol": symbol, "timestamp": datetime.now(), "data": data})

    async def _process_trading_signal(self, symbol: str, timestamp: datetime, data: Any):
        """Process trading signal through the full pipeline"""
        try:
            # 1. Prepare market data
            market_data = self.data_pipeline.prepare_features(data)

            # 2. Get LSTM predictions
            lstm_predictions = await self._get_lstm_predictions(market_data)

            # 3. Get sentiment analysis
            sentiment_scores = await self._get_sentiment_analysis(symbol, timestamp)

            # 4. Prepare state for RL agent
            state = self._prepare_rl_state(
                market_data=market_data,
                lstm_predictions=lstm_predictions,
                sentiment_scores=sentiment_scores,
            )

            # 5. Get RL agent decision
            action, confidence = self.rl_agent.predict(state, deterministic=True)

            # 6. Execute trade if needed
            if action != 0:  # Not HOLD
                trade_result = await self._execute_trade(
                    symbol=symbol, action=action, confidence=confidence, timestamp=timestamp
                )

                # Update performance metrics
                if trade_result["success"]:
                    self.performance_metrics["successful_trades"] += 1
                self.performance_metrics["total_trades"] += 1
                self.performance_metrics["total_pnl"] += trade_result.get("pnl", 0)

            # 7. Log decision
            self._log_trading_decision(
                symbol=symbol,
                timestamp=timestamp,
                action=action,
                confidence=confidence,
                lstm_predictions=lstm_predictions,
                sentiment_scores=sentiment_scores,
            )

        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {str(e)}")
            raise

    async def _get_lstm_predictions(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Get LSTM predictions for multiple horizons"""
        predictions = {}

        for horizon in self.config["prediction_horizons"]:
            pred = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.lstm_predictor.predict, market_data, horizon
            )
            predictions[f"horizon_{horizon}d"] = pred

        return predictions

    async def _get_sentiment_analysis(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        """Get sentiment analysis for the symbol"""
        # In production, this would fetch real news
        # For now, return mock sentiment
        return {
            "overall_sentiment": np.random.uniform(-1, 1),
            "news_count": np.random.randint(0, 10),
            "confidence": np.random.uniform(0.5, 1.0),
        }

    def _prepare_rl_state(
        self,
        market_data: pd.DataFrame,
        lstm_predictions: Dict[str, float],
        sentiment_scores: Dict[str, float],
    ) -> np.ndarray:
        """Prepare state vector for RL agent"""
        # Extract latest market features
        latest_features = market_data.iloc[-1].values

        # Add LSTM predictions
        lstm_features = np.array(list(lstm_predictions.values()))

        # Add sentiment scores
        sentiment_features = np.array(
            [sentiment_scores["overall_sentiment"], sentiment_scores["confidence"]]
        )

        # Combine all features
        state = np.concatenate([latest_features, lstm_features, sentiment_features])

        return state

    async def _execute_trade(
        self, symbol: str, action: int, confidence: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """Execute trade based on agent decision"""
        try:
            # Map action to trade details
            action_map = {
                1: ("BUY", 0.25),
                2: ("BUY", 0.50),
                3: ("BUY", 0.75),
                4: ("BUY", 1.00),
                5: ("SELL", 0.25),
                6: ("SELL", 0.50),
                7: ("SELL", 0.75),
                8: ("SELL", 1.00),
            }

            if action not in action_map:
                return {"success": False, "reason": "Invalid action"}

            side, size_pct = action_map[action]

            # Calculate position size based on risk limit
            position_size = self._calculate_position_size(
                symbol=symbol, size_pct=size_pct, confidence=confidence
            )

            # Execute based on mode
            if self.config["mode"] == "backtest":
                result = self.backtester.place_order(
                    symbol=symbol, side=side, size=position_size, timestamp=timestamp
                )
            else:
                result = await self.data_client.place_order(
                    symbol=symbol, side=side, size=position_size, order_type="MARKET"
                )

            return {
                "success": True,
                "order_id": result.get("order_id"),
                "executed_size": position_size,
                "side": side,
                "pnl": result.get("pnl", 0),
            }

        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return {"success": False, "reason": str(e)}

    def _calculate_position_size(self, symbol: str, size_pct: float, confidence: float) -> int:
        """Calculate position size based on risk management"""
        # Get account balance
        if self.config["mode"] == "backtest":
            balance = self.backtester.get_account_balance()
        else:
            balance = self.config["initial_capital"]  # Simplified

        # Apply risk limit
        max_position_value = balance * self.config["risk_limit"]

        # Adjust by confidence
        position_value = max_position_value * size_pct * confidence

        # Convert to shares (simplified - in production would use current price)
        shares = int(position_value / 100)  # Assuming $100 per share

        return max(1, shares)

    def _log_trading_decision(self, **kwargs):
        """Log trading decision for analysis"""
        log_entry = {
            "timestamp": kwargs["timestamp"].isoformat(),
            "symbol": kwargs["symbol"],
            "action": kwargs["action"],
            "confidence": kwargs["confidence"],
            "lstm_predictions": kwargs["lstm_predictions"],
            "sentiment_scores": kwargs["sentiment_scores"],
        }

        # Save to decision log
        log_path = Path("logs/trading_decisions.jsonl")
        log_path.parent.mkdir(exist_ok=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _generate_backtest_report(self, symbol: str):
        """Generate backtest report for a symbol"""
        if not self.backtester:
            return

        report = self.backtester.get_performance_summary()
        report["symbol"] = symbol
        report["config"] = self.config

        # Save report
        report_path = Path(
            f'reports/backtest_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Backtest report saved to {report_path}")

    async def stop(self):
        """Stop the trading system gracefully"""
        logger.info("Stopping trading system...")

        self.is_running = False
        self.shutdown_event.set()

        # Stop components
        if self.health_monitor:
            await self.health_monitor.stop()

        if self.data_pipeline:
            await self.data_pipeline.stop()

        if self.data_client:
            await self.data_client.disconnect()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Generate final report
        self._generate_performance_report()

        logger.info("Trading system stopped")

    def _generate_performance_report(self):
        """Generate final performance report"""
        runtime = datetime.now() - self.performance_metrics["start_time"]

        report = {
            "runtime": str(runtime),
            "total_trades": self.performance_metrics["total_trades"],
            "successful_trades": self.performance_metrics["successful_trades"],
            "success_rate": (
                self.performance_metrics["successful_trades"]
                / max(1, self.performance_metrics["total_trades"])
            ),
            "total_pnl": self.performance_metrics["total_pnl"],
            "errors": self.performance_metrics["errors"],
        }

        # Save report
        report_path = Path(f'reports/performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report saved to {report_path}")


async def main():
    """Main entry point"""
    controller = MainController()

    try:
        await controller.initialize_components()
        await controller.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await controller.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(main())
