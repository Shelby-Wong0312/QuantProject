"""
Dashboard Data Connector
儀表板數據連接器
Cloud DE - Task DE-403
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List
import websocket
import threading
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.data.realtime_collector import RealtimeDataCollector
    from src.risk.risk_manager_enhanced import EnhancedRiskManager
    from src.core.paper_trading import PaperTradingSimulator
except ImportError:
    # Create dummy classes if modules not available
    class RealtimeDataCollector:
        pass

    class EnhancedRiskManager:
        pass

    class PaperTradingSimulator:
        pass


logger = logging.getLogger(__name__)


class DashboardDataConnector:
    """
    Data connector for dashboard
    Connects to various data sources and provides unified interface
    """

    def __init__(
        self,
        data_dir: str = "data",
        reports_dir: str = "reports",
        use_websocket: bool = False,
    ):
        """
        Initialize data connector

        Args:
            data_dir: Data directory path
            reports_dir: Reports directory path
            use_websocket: Enable WebSocket for real-time updates
        """
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.use_websocket = use_websocket

        # Data caches
        self.portfolio_cache = {}
        self.market_data_cache = {}
        self.risk_metrics_cache = {}

        # WebSocket connection
        self.ws = None
        self.ws_thread = None

        # Components
        self.data_collector = None
        self.risk_manager = None
        self.paper_trader = None

        # Initialize components
        self._initialize_components()

        logger.info("Dashboard Data Connector initialized")

    def _initialize_components(self):
        """Initialize trading system components"""
        try:
            # Initialize data collector
            self.data_collector = RealtimeDataCollector(
                ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            )

            # Initialize risk manager
            self.risk_manager = EnhancedRiskManager(initial_capital=100000)

            # Initialize paper trader
            self.paper_trader = PaperTradingSimulator(initial_balance=100000)

            logger.info("Trading components initialized")

        except Exception as e:
            logger.warning(f"Could not initialize components: {e}")

    def connect_websocket(self, url: str = "ws://localhost:8080"):
        """
        Connect to WebSocket for real-time updates

        Args:
            url: WebSocket URL
        """
        if not self.use_websocket:
            return

        def on_message(ws, message):
            """Handle WebSocket message"""
            try:
                json.loads(message)
                self._process_websocket_data(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")

        def on_error(ws, error):
            """Handle WebSocket error"""
            logger.error(f"WebSocket error: {error}")

        def on_close(ws):
            """Handle WebSocket close"""
            logger.info("WebSocket connection closed")

        def on_open(ws):
            """Handle WebSocket open"""
            logger.info("WebSocket connection opened")
            # Subscribe to data streams
            ws.send(
                json.dumps(
                    {
                        "action": "subscribe",
                        "channels": ["portfolio", "market_data", "risk_metrics"],
                    }
                )
            )

        try:
            self.ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )

            # Run WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()

            logger.info(f"WebSocket connected to {url}")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    def _process_websocket_data(self, data: Dict):
        """
        Process incoming WebSocket data

        Args:
            data: WebSocket data
        """
        channel = data.get("channel")
        payload = data.get("payload", {})

        if channel == "portfolio":
            self.portfolio_cache.update(payload)
        elif channel == "market_data":
            self.market_data_cache.update(payload)
        elif channel == "risk_metrics":
            self.risk_metrics_cache.update(payload)

    def get_portfolio_data(self) -> Dict:
        """
        Get current portfolio data

        Returns:
            Portfolio data dictionary
        """
        # Check cache first
        if self.portfolio_cache:
            return self.portfolio_cache

        # Try to load from paper trader
        if self.paper_trader:
            try:
                metrics = self.paper_trader.get_performance_metrics()
                positions = self.paper_trader.positions

                portfolio_data = {
                    "account": {
                        "initial_balance": self.paper_trader.account.initial_balance,
                        "cash_balance": self.paper_trader.account.cash_balance,
                        "portfolio_value": metrics["portfolio_value"],
                        "total_pnl": metrics["total_pnl"],
                        "total_commission": metrics["total_commission"],
                    },
                    "positions": {
                        symbol: {
                            "quantity": pos.quantity,
                            "avg_price": pos.avg_price,
                            "current_price": pos.current_price,
                            "unrealized_pnl": pos.unrealized_pnl,
                            "market_value": pos.market_value,
                        }
                        for symbol, pos in positions.items()
                    },
                    "performance": {
                        "total_return": metrics["total_return"],
                        "win_rate": metrics["win_rate"],
                        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                        "max_drawdown": metrics.get("max_drawdown", 0),
                        "total_trades": metrics["total_trades"],
                    },
                }

                self.portfolio_cache = portfolio_data
                return portfolio_data

            except Exception as e:
                logger.error(f"Error getting portfolio data: {e}")

        # Try to load from file
        state_file = self.reports_dir / "paper_trading_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                return json.load(f)

        # Return default data
        return self._get_default_portfolio_data()

    def get_historical_performance(
        self, period: str = "1M", interval: str = "1D"
    ) -> pd.DataFrame:
        """
        Get historical performance data

        Args:
            period: Time period (1D, 1W, 1M, 3M, YTD)
            interval: Data interval (1m, 5m, 15m, 1H, 1D)

        Returns:
            DataFrame with historical data
        """
        # Define period mappings
        period_days = {
            "1D": 1,
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
        }

        days = period_days.get(period, 30)

        # Try to load from data collector
        if self.data_collector:
            try:
                # Get minute bars for each symbol
                all_data = []
                for symbol in self.data_collector.symbols:
                    bars = self.data_collector.get_minute_bars(symbol, days * 24 * 60)
                    if not bars.empty:
                        all_data.append(bars)

                if all_data:
                    # Combine and aggregate
                    combined = pd.concat(all_data)
                    return self._aggregate_performance_data(combined, interval)

            except Exception as e:
                logger.error(f"Error getting historical data: {e}")

        # Generate synthetic data
        return self._generate_synthetic_performance(days, interval)

    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics

        Returns:
            Risk metrics dictionary
        """
        # Check cache first
        if self.risk_metrics_cache:
            return self.risk_metrics_cache

        # Try to get from risk manager
        if self.risk_manager:
            try:
                self.risk_manager.get_risk_report()

                risk_metrics = {
                    "risk_score": 45,  # Default
                    "var_95": -5000,
                    "cvar_95": -7500,
                    "leverage": 1.5,
                    "concentration_risk": 0.25,
                    "max_drawdown": report["current_status"]["current_drawdown"],
                    "daily_pnl": report["current_status"]["daily_pnl"],
                    "active_alerts": report["current_status"]["active_alerts"],
                }

                self.risk_metrics_cache = risk_metrics
                return risk_metrics

            except Exception as e:
                logger.error(f"Error getting risk metrics: {e}")

        # Try to load from file
        report_file = self.reports_dir / "stress_test_report.json"
        if report_file.exists():
            with open(report_file, "r") as f:
                json.load(f)
                return self._parse_stress_test_report(data)

        # Return default metrics
        return self._get_default_risk_metrics()

    def get_recent_trades(self, limit: int = 50) -> pd.DataFrame:
        """
        Get recent trades

        Args:
            limit: Maximum number of trades

        Returns:
            DataFrame with recent trades
        """
        trades = []

        # Try to get from paper trader
        if self.paper_trader:
            try:
                for trade in self.paper_trader.trade_history[-limit:]:
                    trades.append(
                        {
                            "time": trade.get("timestamp", datetime.now()),
                            "symbol": trade.get("symbol", "N/A"),
                            "side": trade.get("side", "N/A"),
                            "quantity": trade.get("quantity", 0),
                            "price": trade.get("price", 0),
                            "commission": trade.get("commission", 0),
                            "pnl": trade.get("pnl", 0),
                        }
                    )

            except Exception as e:
                logger.error(f"Error getting trades: {e}")

        # If no trades, generate sample data
        if not trades:
            trades = self._generate_sample_trades(limit)

        return pd.DataFrame(trades)

    def get_alerts(self) -> List[Dict]:
        """
        Get active risk alerts

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Get from risk manager
        if self.risk_manager:
            try:
                for alert in self.risk_manager.alerts[-10:]:  # Last 10 alerts
                    alerts.append(
                        {
                            "timestamp": alert.timestamp,
                            "level": alert.level,
                            "category": alert.category,
                            "message": alert.message,
                            "action_required": alert.action_required,
                        }
                    )

            except Exception as e:
                logger.error(f"Error getting alerts: {e}")

        return alerts

    def _aggregate_performance_data(
        self, df: pd.DataFrame, interval: str
    ) -> pd.DataFrame:
        """Aggregate performance data to specified interval"""
        # Resample based on interval
        resample_map = {"1m": "1T", "5m": "5T", "15m": "15T", "1H": "1H", "1D": "1D"}

        rule = resample_map.get(interval, "1D")

        aggregated = df.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        return aggregated

    def _generate_synthetic_performance(self, days: int, interval: str) -> pd.DataFrame:
        """Generate synthetic performance data"""
        # Calculate number of points
        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "1D": 1440}

        minutes = interval_minutes.get(interval, 1440)
        points = (days * 1440) // minutes

        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=points, freq=f"{minutes}T")

        # Generate returns
        returns = np.random.normal(0.0001, 0.02, points)
        cumulative = (1 + returns).cumprod()
        values = 100000 * cumulative

        df = pd.DataFrame(
            {
                "date": dates,
                "value": values,
                "return": returns,
                "volume": np.random.randint(1000000, 5000000, points),
            }
        )

        return df

    def _generate_sample_trades(self, limit: int) -> List[Dict]:
        """Generate sample trades for demonstration"""
        trades = []
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        for i in range(limit):
            trade_time = datetime.now() - timedelta(hours=i * 2)
            side = np.random.choice(["BUY", "SELL"])
            quantity = np.random.randint(10, 100)
            price = np.random.uniform(100, 400)

            trades.append(
                {
                    "time": trade_time,
                    "symbol": np.random.choice(symbols),
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "commission": quantity * price * 0.001,
                    "pnl": np.random.uniform(-500, 1000) if side == "SELL" else 0,
                }
            )

        return trades

    def _get_default_portfolio_data(self) -> Dict:
        """Get default portfolio data"""
        return {
            "account": {
                "initial_balance": 100000,
                "cash_balance": 45000,
                "portfolio_value": 112500,
                "total_pnl": 12500,
                "total_commission": 125,
            },
            "positions": {
                "AAPL": {
                    "quantity": 100,
                    "avg_price": 180,
                    "current_price": 185,
                    "unrealized_pnl": 500,
                    "market_value": 18500,
                },
                "GOOGL": {
                    "quantity": 50,
                    "avg_price": 140,
                    "current_price": 142,
                    "unrealized_pnl": 100,
                    "market_value": 7100,
                },
            },
            "performance": {
                "total_return": 0.125,
                "win_rate": 0.65,
                "sharpe_ratio": 1.35,
                "max_drawdown": -0.08,
                "total_trades": 150,
            },
        }

    def _get_default_risk_metrics(self) -> Dict:
        """Get default risk metrics"""
        return {
            "risk_score": 45,
            "var_95": -5000,
            "cvar_95": -7500,
            "leverage": 1.5,
            "concentration_risk": 0.25,
            "max_drawdown": -0.08,
            "daily_pnl": 250,
            "active_alerts": 0,
        }

    def _parse_stress_test_report(self, report: Dict) -> Dict:
        """Parse stress test report for risk metrics"""
        summary = report.get("summary", {})

        return {
            "risk_score": 45,  # Calculate based on results
            "var_95": summary.get("worst_case_impact", -0.1) * 100000,
            "cvar_95": summary.get("worst_case_impact", -0.1) * 100000 * 1.5,
            "leverage": 1.5,
            "concentration_risk": 0.25,
            "max_drawdown": summary.get("worst_case_impact", -0.1),
            "daily_pnl": 0,
            "active_alerts": 0,
        }

    def close(self):
        """Close connections and cleanup"""
        if self.ws:
            self.ws.close()

        logger.info("Data connector closed")


if __name__ == "__main__":
    # Test data connector
    connector = DashboardDataConnector()

    print("Testing Dashboard Data Connector...")
    print("=" * 50)

    # Test portfolio data
    portfolio = connector.get_portfolio_data()
    print(f"Portfolio Value: ${portfolio['account']['portfolio_value']:,.0f}")
    print(f"Total P&L: ${portfolio['account']['total_pnl']:,.0f}")

    # Test risk metrics
    risk = connector.get_risk_metrics()
    print(f"\nRisk Score: {risk['risk_score']}")
    print(f"VaR (95%): ${risk['var_95']:,.0f}")

    # Test recent trades
    trades = connector.get_recent_trades(5)
    print(f"\nRecent Trades: {len(trades)} trades")

    print("\nData Connector Test Complete!")
