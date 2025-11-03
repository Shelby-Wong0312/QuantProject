"""
Live Automated Trading System - Full Market Coverage
實時自動交易系統 - 全市場監控版本
"""

import asyncio
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

# Set API credentials
# os.environ['CAPITAL_API_KEY'] removed - use .env file
# os.environ['CAPITAL_IDENTIFIER'] removed - use .env file
# os.environ['CAPITAL_API_PASSWORD'] removed - use .env file
os.environ["CAPITAL_DEMO_MODE"] = "True"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.connectors.capital_com_api import CapitalComAPI
from src.risk.risk_manager_enhanced import EnhancedRiskManager
from src.signals.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/live_trading_full.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class FullMarketTradingSystem:
    """全市場自動交易系統 - 監控所有4,215支股票"""

    def __init__(self):
        self.api = None
        self.risk_manager = None
        self.signal_generator = None
        self.active_positions = {}
        self.trade_history = []
        self.running = False
        self.total_trades = 0
        self.profitable_trades = 0

        # Session management
        self.session_created = None
        self.last_ping = None
        self.session_expiry = 600  # 10 minutes in seconds
        self.ping_interval = 300  # Ping every 5 minutes

        # API rate limiting
        self.request_count = 0
        self.request_window_start = datetime.now()
        self.max_requests_per_second = 10
        self.demo_hourly_limit = 1000
        self.demo_request_count = 0
        self.demo_hour_start = datetime.now()

        # Load all available stocks
        self.all_symbols = self.load_all_symbols()
        # Limit to 40 symbols for WebSocket (Capital.com limit)
        self.monitored_symbols = set(self.all_symbols[:40])
        self.top_movers = []  # Today's top movers
        self.watchlist = []  # High priority stocks

        # Trading parameters
        self.max_positions = 20  # Increased from 5 to 20
        self.position_size_pct = 0.05  # 5% of portfolio per position
        self.min_volume = 1000000  # Minimum daily volume
        self.scan_interval = 60  # Full scan every 60 seconds
        self.quick_scan_interval = 5  # Quick scan top movers every 5 seconds

        # Performance tracking
        self.daily_pnl = 0
        self.total_pnl = 0
        self.scanned_stocks = 0
        self.signals_generated = 0

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)

    def load_all_symbols(self) -> List[str]:
        """載入所有可交易股票"""
        []
        try:
            conn = sqlite3.connect("data/quant_trading.db")
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM stocks ORDER BY symbol")
            [row[0] for row in cursor.fetchall()]
            conn.close()
            # Remove problematic symbols
            problem_symbols = ["JNPR", "N", "V", "K"]  # Known issues
            [s for s in symbols if s not in problem_symbols]

            logger.info(f"Loaded {len(symbols)} symbols from database")
        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
            # Fallback to default list
            [
                "AAPL",
                "MSFT",
                "GOOGL",
                "TSLA",
                "NVDA",
                "META",
                "AMZN",
                "NFLX",
                "AMD",
                "INTC",
            ]

        return symbols

    async def initialize(self):
        """初始化系統組件"""
        print("\n[INIT] Initializing Full Market Trading System...")
        print(f"[INIT] Total stocks to monitor: {len(self.all_symbols)}")

        # 1. Initialize API
        print("[INIT] Connecting to Capital.com API...")
        self.api = CapitalComAPI()
        if not self.api.authenticate():
            print("[WARN] Capital.com API not connected - using YFinance backup")
        else:
            print("[OK] Connected to Capital.com")

        # 2. Initialize Risk Manager
        print("[INIT] Setting up Risk Manager...")
        self.risk_manager = EnhancedRiskManager(
            initial_capital=140370.87,
            max_daily_loss=0.02,
            max_position_loss=0.01,
            max_drawdown=0.10,
        )
        print("[OK] Risk Manager configured")

        # 3. Initialize Signal Generator
        print("[INIT] Loading Signal Generator...")
        self.signal_generator = SignalGenerator()
        print("[OK] Signal Generator ready")

        # 4. Initialize database
        self.init_database()
        print("[OK] Database connected")

        # 5. Initial market scan
        print("[INIT] Performing initial market scan...")
        await self.initial_market_scan()
        print(f"[OK] Initial scan complete - {len(self.watchlist)} stocks in watchlist")

        print("\n[READY] System initialized successfully!")
        print(f"[READY] Monitoring {len(self.all_symbols)} stocks")
        print(f"[READY] Max positions: {self.max_positions}")
        print(f"[READY] Position size: {self.position_size_pct*100}% per trade")
        return True

    def init_database(self):
        """初始化交易數據庫"""
        conn = sqlite3.connect("data/live_trades_full.db")
        cursor = conn.cursor()

        # Trades table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                quantity INTEGER,
                price REAL,
                total_value REAL,
                pnl REAL,
                status TEXT
            )
        """
        )

        # Signals table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                signal_type TEXT,
                strength REAL,
                price REAL,
                volume INTEGER,
                rsi REAL,
                macd REAL
            )
        """
        )

        # Performance table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                profitable_trades INTEGER,
                total_pnl REAL,
                stocks_scanned INTEGER,
                signals_generated INTEGER
            )
        """
        )

        conn.commit()
        conn.close()

    async def initial_market_scan(self):
        """初始市場掃描 - 找出最活躍的股票"""
        print(f"[SCAN] Scanning {len(self.all_symbols)} stocks...")

        active_stocks = []
        batch_size = 100

        for i in range(0, len(self.all_symbols), batch_size):
            batch = self.all_symbols[i : i + batch_size]

            # Use ThreadPoolExecutor for parallel processing
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(self.executor, self.scan_batch, batch)

            active_stocks.extend(results)

            # Progress update
            if i % 500 == 0:
                print(f"[SCAN] Progress: {i}/{len(self.all_symbols)} stocks scanned")

        # Sort by volume and volatility
        active_stocks.sort(key=lambda x: x["score"], reverse=True)

        # Top 100 most active stocks for frequent monitoring
        self.watchlist = [s["symbol"] for s in active_stocks[:100]]

        # Top 20 for immediate monitoring
        self.monitored_symbols = set(self.watchlist[:20])

        print(f"[SCAN] Found {len(active_stocks)} active stocks")
        print(f"[SCAN] Watchlist: {len(self.watchlist)} stocks")
        print(f"[SCAN] Priority monitoring: {len(self.monitored_symbols)} stocks")

    def scan_batch(self, symbols: List[str]) -> List[Dict]:
        """掃描一批股票"""
        results = []

        for symbol in symbols:
            try:
                # Get basic data using yfinance
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Check if stock meets criteria
                volume = info.get("volume", 0)
                if volume < self.min_volume:
                    continue

                # Calculate activity score
                price = info.get("currentPrice", 0)
                if price < 1:  # Skip penny stocks
                    continue

                # Get recent volatility
                hist = ticker.history(period="5d")
                if not hist.empty:
                    volatility = hist["Close"].pct_change().std()

                    # Activity score = volume * volatility * price
                    score = volume * volatility * min(price / 100, 1)

                    results.append(
                        {
                            "symbol": symbol,
                            "price": price,
                            "volume": volume,
                            "volatility": volatility,
                            "score": score,
                        }
                    )

            except Exception:
                # Skip stocks that cause errors
                pass

        return results

    async def scan_market(self):
        """定期市場掃描"""
        self.scanned_stocks = 0
        {}

        # Priority 1: Check existing positions
        for symbol in self.active_positions.keys():
            signal = await self.analyze_symbol_async(symbol)
            if signal:
                signals[symbol] = signal
                self.scanned_stocks += 1

        # Priority 2: Scan watchlist
        for symbol in self.watchlist[:50]:  # Top 50 from watchlist
            if symbol not in self.active_positions:
                signal = await self.analyze_symbol_async(symbol)
                if signal and signal != "HOLD":
                    signals[symbol] = signal
                    self.signals_generated += 1
                self.scanned_stocks += 1

        # Priority 3: Random sampling from all stocks
        import random

        sample_size = min(50, len(self.all_symbols) - len(self.watchlist))
        random_stocks = random.sample(
            [s for s in self.all_symbols if s not in self.watchlist], sample_size
        )

        for symbol in random_stocks:
            signal = await self.analyze_symbol_async(symbol)
            if signal and signal != "HOLD":
                signals[symbol] = signal
                self.signals_generated += 1
            self.scanned_stocks += 1

        logger.info(
            f"[SCAN] Scanned {self.scanned_stocks} stocks, generated {len(signals)} signals"
        )
        return signals

    async def analyze_symbol_async(self, symbol: str) -> Optional[str]:
        """異步分析股票"""
        try:
            ticker = yf.Ticker(symbol)

            # Get recent data
            hist = ticker.history(period="20d")
            if hist.empty or len(hist) < 20:
                return None

            # Calculate indicators
            close_prices = hist["Close"].values
            volumes = hist["Volume"].values

            # Simple RSI
            deltas = np.diff(close_prices)
            gains = deltas[deltas > 0].sum()
            losses = -deltas[deltas < 0].sum()
            rs = gains / (losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Simple MACD
            ema12 = pd.Series(close_prices).ewm(span=12).mean().iloc[-1]
            ema26 = pd.Series(close_prices).ewm(span=26).mean().iloc[-1]
            macd = ema12 - ema26

            # Volume surge
            avg_volume = volumes[:-1].mean()
            current_volume = volumes[-1]
            volume_ratio = current_volume / (avg_volume + 1)

            # Generate signal
            current_price = close_prices[-1]

            # Buy signals
            if rsi < 30 and volume_ratio > 1.5:
                return "BUY"
            elif macd > 0 and rsi < 50 and volume_ratio > 1.2:
                return "BUY"

            # Sell signals
            elif rsi > 70:
                return "SELL"
            elif macd < -0.5 and rsi > 60:
                return "SELL"

            # Check existing positions for exit
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                pnl_pct = (current_price - position["entry_price"]) / position[
                    "entry_price"
                ]

                if pnl_pct <= -0.05:  # Stop loss
                    return "SELL"
                elif pnl_pct >= 0.10:  # Take profit
                    return "SELL"

            return "HOLD"

        except Exception:
            return None

    async def execute_trade(self, symbol: str, action: str):
        """執行交易 (with Capital.com API rate limits)"""
        try:
            # Capital.com API rate limits:
            # - Max 10 requests per second
            # - Position/Order creation: 0.1 second minimum interval
            # - Demo account: 1000 requests per hour
            await asyncio.sleep(0.1)  # Minimum 0.1s between position/order requests

            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get("currentPrice", 0)
            if not current_price:
                return

            # Calculate position size
            portfolio_value = 140370.87  # Should get from API
            position_value = portfolio_value * self.position_size_pct
            shares = int(position_value / current_price)

            if shares <= 0:
                return

            # Risk check
            if not self.risk_manager.check_trade_allowed(
                symbol=symbol, quantity=shares, price=current_price
            ):
                logger.warning(f"Trade rejected by risk manager: {symbol} {action}")
                return

            if action == "BUY":
                if len(self.active_positions) >= self.max_positions:
                    logger.info(f"Max positions ({self.max_positions}) reached")
                    return

                # Record position
                self.active_positions[symbol] = {
                    "quantity": shares,
                    "entry_price": current_price,
                    "entry_time": datetime.now(),
                }
                self.total_trades += 1

                # Save trade
                self.save_trade(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": shares,
                        "price": current_price,
                        "total_value": shares * current_price,
                        "status": "EXECUTED",
                    }
                )

                logger.info(
                    f"[TRADE] Bought {shares} shares of {symbol} at ${current_price:.2f}"
                )

            elif action == "SELL" and symbol in self.active_positions:
                position = self.active_positions[symbol]

                # Calculate P&L
                pnl = (current_price - position["entry_price"]) * position["quantity"]
                self.daily_pnl += pnl
                self.total_pnl += pnl

                if pnl > 0:
                    self.profitable_trades += 1

                # Remove position
                del self.active_positions[symbol]

                # Save trade
                self.save_trade(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": position["quantity"],
                        "price": current_price,
                        "total_value": position["quantity"] * current_price,
                        "pnl": pnl,
                        "status": "EXECUTED",
                    }
                )

                logger.info(
                    f"[TRADE] Sold {position['quantity']} shares of {symbol} at ${current_price:.2f}, P&L: ${pnl:.2f}"
                )

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")

    def save_trade(self, trade_data: Dict):
        """保存交易記錄"""
        conn = sqlite3.connect("data/live_trades_full.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (timestamp, symbol, action, quantity, price, total_value, pnl, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade_data["timestamp"],
                trade_data["symbol"],
                trade_data["action"],
                trade_data["quantity"],
                trade_data["price"],
                trade_data["total_value"],
                trade_data.get("pnl", 0),
                trade_data["status"],
            ),
        )
        conn.commit()
        conn.close()

    def display_status(self):
        """顯示系統狀態"""
        print("\n" + "=" * 80)
        print(f"[STATUS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Portfolio summary
        print("\n📊 PORTFOLIO SUMMARY")
        print(f"Active Positions: {len(self.active_positions)}/{self.max_positions}")
        print(f"Today's P&L: ${self.daily_pnl:+,.2f}")
        print(f"Total P&L: ${self.total_pnl:+,.2f}")

        # Positions
        if self.active_positions:
            print("\n📈 OPEN POSITIONS:")
            for symbol, position in list(self.active_positions.items())[
                :10
            ]:  # Show top 10
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.info.get(
                        "currentPrice", position["entry_price"]
                    )
                    pnl = (current_price - position["entry_price"]) * position[
                        "quantity"
                    ]
                    pnl_pct = (
                        (current_price - position["entry_price"])
                        / position["entry_price"]
                        * 100
                    )
                    print(
                        f"  {symbol:6} {position['quantity']:4} shares | Entry: ${position['entry_price']:.2f} | Current: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)"
                    )
                except Exception:
                    pass

        # Statistics
        win_rate = (
            (self.profitable_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )

        print("\n📊 STATISTICS:")
        print(f"Stocks Monitored: {len(self.all_symbols)}")
        print(f"Stocks Scanned (last cycle): {self.scanned_stocks}")
        print(f"Signals Generated: {self.signals_generated}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print("-" * 80)

    async def maintain_session(self):
        """維護API session - 每5分鐘ping一次"""
        if not self.api:
            return

        # Check if we need to ping
        if (
            self.last_ping is None
            or (datetime.now() - self.last_ping).seconds > self.ping_interval
        ):
            try:
                # Ping the API to keep session alive
                self.api.ping()
                self.last_ping = datetime.now()
                logger.info("Session pinged successfully")
            except Exception as e:
                logger.error(f"Failed to ping session: {e}")
                # Try to re-authenticate
                self.api.authenticate()
                self.session_created = datetime.now()
                self.last_ping = datetime.now()

    async def check_rate_limits(self):
        """檢查並遵守API速率限制"""
        # Check per-second limit
        if (datetime.now() - self.request_window_start).seconds >= 1:
            self.request_count = 0
            self.request_window_start = datetime.now()

        if self.request_count >= self.max_requests_per_second:
            await asyncio.sleep(0.1)  # Wait before next request

        # Check demo hourly limit
        if (datetime.now() - self.demo_hour_start).seconds >= 3600:
            self.demo_request_count = 0
            self.demo_hour_start = datetime.now()

        if self.demo_request_count >= self.demo_hourly_limit:
            logger.warning("Demo hourly limit reached, waiting...")
            await asyncio.sleep(60)  # Wait 1 minute

    async def run(self):
        """主交易循環"""
        self.running = True
        logger.info(
            "Starting automated trading - monitoring top 40 stocks (WebSocket limit)..."
        )

        scan_counter = 0

        while self.running:
            try:
                scan_counter += 1

                # Maintain session
                await self.maintain_session()

                # Check rate limits
                await self.check_rate_limits()

                # Full market scan every 12 cycles (12 minutes)
                if scan_counter % 12 == 0:
                    print("\n[SCAN] Performing full market scan...")
                    await self.initial_market_scan()

                # Regular scan
                print(f"\n[SCAN] Scanning market... (Cycle {scan_counter})")
                await self.scan_market()

                # Execute trades
                for symbol, signal in signals.items():
                    if signal in ["BUY", "SELL"]:
                        await self.execute_trade(symbol, signal)

                # Display status
                self.display_status()

                # Save daily performance
                if datetime.now().hour == 16 and datetime.now().minute == 0:
                    self.save_daily_performance()
                    self.daily_pnl = 0  # Reset daily P&L

                # Wait for next cycle
                await asyncio.sleep(self.scan_interval)

            except KeyboardInterrupt:
                logger.info("Stopping trading system...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

    def save_daily_performance(self):
        """保存每日績效"""
        conn = sqlite3.connect("data/live_trades_full.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO performance 
            (date, total_trades, profitable_trades, total_pnl, stocks_scanned, signals_generated)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().date().isoformat(),
                self.total_trades,
                self.profitable_trades,
                self.total_pnl,
                self.scanned_stocks,
                self.signals_generated,
            ),
        )
        conn.commit()
        conn.close()

    async def shutdown(self):
        """關閉系統"""
        logger.info("Shutting down trading system...")

        # Close all positions
        for symbol in list(self.active_positions.keys()):
            logger.info(f"Closing position: {symbol}")
            await self.execute_trade(symbol, "SELL")

        # Save final performance
        self.save_daily_performance()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("System shutdown complete.")


async def main():
    """主程序"""
    print("\n" + "=" * 60)
    print("     FULL MARKET AUTOMATED TRADING SYSTEM")
    print("          Monitoring 4,215 Stocks")
    print("-" * 60)
    print("  Configuration:")
    print("  - Max Positions: 20")
    print("  - Position Size: 5% per trade")
    print("  - Stop Loss: 5%")
    print("  - Take Profit: 10%")
    print("  - Scan Interval: 60 seconds")
    print("-" * 60)
    print("       Press Ctrl+C to stop trading")
    print("=" * 60 + "\n")

    system = FullMarketTradingSystem()

    try:
        # Initialize system
        if await system.initialize():
            # Start trading
            await system.run()
    except KeyboardInterrupt:
        print("\n[STOPPING] User requested shutdown...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)

    # Run the trading system
    asyncio.run(main())
