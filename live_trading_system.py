"""
Live Automated Trading System
實時自動交易系統 - Production Ready
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.connectors.capital_com_api import CapitalComAPI
from src.risk.risk_manager_enhanced import EnhancedRiskManager
from src.signals.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """實時自動交易系統"""
    
    def __init__(self):
        self.api = None
        self.risk_manager = None
        self.signal_generator = None
        self.active_positions = {}
        self.trade_history = []
        self.running = False
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Trading parameters
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.position_size = 100  # shares per trade
        self.max_positions = 5
        self.check_interval = 60  # seconds
        
    async def initialize(self):
        """初始化系統組件"""
        print("\n[INIT] Initializing Trading System...")
        
        # 1. Initialize API
        print("[INIT] Connecting to Capital.com API...")
        self.api = CapitalComAPI()
        if not self.api.authenticate():
            raise Exception("Failed to connect to Capital.com API")
        print(f"[OK] Connected - Account: {self.api.account_id}")
        
        # 2. Get account info
        account_info = self.api.get_account_info()
        if account_info:
            balance = account_info.get('balance', 0)
            print(f"[OK] Account Balance: ${balance:,.2f}")
        
        # 3. Initialize Risk Manager
        print("[INIT] Setting up Risk Manager...")
        self.risk_manager = EnhancedRiskManager(
            initial_capital=140370.87,
            max_daily_loss=0.02,
            max_position_loss=0.01,
            max_drawdown=0.10
        )
        print("[OK] Risk Manager configured")
        
        # 4. Initialize Signal Generator
        print("[INIT] Loading Signal Generator...")
        self.signal_generator = SignalGenerator()
        print("[OK] Signal Generator ready")
        
        # 5. Initialize database
        self.init_database()
        print("[OK] Database connected")
        
        print("\n[READY] System initialized successfully!")
        return True
    
    def init_database(self):
        """初始化交易數據庫"""
        conn = sqlite3.connect('data/live_trades.db')
        cursor = conn.cursor()
        cursor.execute('''
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
        ''')
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data: Dict):
        """保存交易記錄"""
        conn = sqlite3.connect('data/live_trades.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, action, quantity, price, total_value, pnl, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['timestamp'],
            trade_data['symbol'],
            trade_data['action'],
            trade_data['quantity'],
            trade_data['price'],
            trade_data['total_value'],
            trade_data.get('pnl', 0),
            trade_data['status']
        ))
        conn.commit()
        conn.close()
    
    async def check_market_conditions(self) -> bool:
        """檢查市場條件是否適合交易"""
        # Check if market is open
        now = datetime.now()
        hour = now.hour
        
        # US market hours (9:30 AM - 4:00 PM EST)
        # Convert to your timezone as needed
        if hour < 9 or hour > 16:
            return False
        
        # Check volatility
        # Add volatility check logic here
        
        return True
    
    async def generate_signals(self) -> Dict[str, str]:
        """生成交易信號"""
        signals = {}
        
        for symbol in self.symbols:
            try:
                # Get current price
                price = self.api.get_market_price(symbol)
                if not price:
                    continue
                
                # Generate signal (simplified for demo)
                # In production, use ML models or technical indicators
                signal = self.analyze_symbol(symbol, price)
                signals[symbol] = signal
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def analyze_symbol(self, symbol: str, current_price: float) -> str:
        """分析個股並生成信號"""
        # Simplified logic - replace with real strategy
        import random
        
        # Random signal for demo (replace with real analysis)
        rand = random.random()
        if rand < 0.1:  # 10% chance to buy
            return 'BUY'
        elif rand < 0.15:  # 5% chance to sell
            return 'SELL'
        else:
            return 'HOLD'
    
    async def execute_trade(self, symbol: str, action: str):
        """執行交易"""
        try:
            current_price = self.api.get_market_price(symbol)
            if not current_price:
                return
            
            # Risk check
            if not self.risk_manager.check_trade_allowed(
                symbol=symbol,
                quantity=self.position_size,
                price=current_price
            ):
                logger.warning(f"Trade rejected by risk manager: {symbol} {action}")
                return
            
            # Execute order
            if action == 'BUY':
                if len(self.active_positions) >= self.max_positions:
                    logger.info(f"Max positions reached, skipping {symbol}")
                    return
                
                # Place buy order
                order = self.api.place_order(
                    symbol=symbol,
                    direction='BUY',
                    size=self.position_size,
                    order_type='MARKET'
                )
                
                if order and order.get('status') == 'FILLED':
                    self.active_positions[symbol] = {
                        'quantity': self.position_size,
                        'entry_price': current_price,
                        'entry_time': datetime.now()
                    }
                    self.total_trades += 1
                    
                    # Save trade
                    self.save_trade({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self.position_size,
                        'price': current_price,
                        'total_value': self.position_size * current_price,
                        'status': 'FILLED'
                    })
                    
                    logger.info(f"[TRADE] Bought {self.position_size} shares of {symbol} at ${current_price:.2f}")
            
            elif action == 'SELL' and symbol in self.active_positions:
                # Place sell order
                position = self.active_positions[symbol]
                
                order = self.api.place_order(
                    symbol=symbol,
                    direction='SELL',
                    size=position['quantity'],
                    order_type='MARKET'
                )
                
                if order and order.get('status') == 'FILLED':
                    # Calculate P&L
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    if pnl > 0:
                        self.profitable_trades += 1
                    
                    # Remove from active positions
                    del self.active_positions[symbol]
                    
                    # Save trade
                    self.save_trade({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['quantity'],
                        'price': current_price,
                        'total_value': position['quantity'] * current_price,
                        'pnl': pnl,
                        'status': 'FILLED'
                    })
                    
                    logger.info(f"[TRADE] Sold {position['quantity']} shares of {symbol} at ${current_price:.2f}, P&L: ${pnl:.2f}")
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def monitor_positions(self):
        """監控現有持倉"""
        for symbol, position in list(self.active_positions.items()):
            try:
                current_price = self.api.get_market_price(symbol)
                if not current_price:
                    continue
                
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check stop loss (5%)
                if pnl_pct <= -0.05:
                    logger.warning(f"[STOP LOSS] Triggered for {symbol}")
                    await self.execute_trade(symbol, 'SELL')
                
                # Check take profit (10%)
                elif pnl_pct >= 0.10:
                    logger.info(f"[TAKE PROFIT] Triggered for {symbol}")
                    await self.execute_trade(symbol, 'SELL')
                
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
    
    def display_status(self):
        """顯示系統狀態"""
        print("\n" + "="*60)
        print(f"[STATUS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Account info
        account_info = self.api.get_account_info()
        if account_info:
            print(f"Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"Equity: ${account_info.get('equity', 0):,.2f}")
        
        # Positions
        print(f"\nActive Positions: {len(self.active_positions)}/{self.max_positions}")
        for symbol, position in self.active_positions.items():
            current_price = self.api.get_market_price(symbol)
            if current_price:
                pnl = (current_price - position['entry_price']) * position['quantity']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                print(f"  {symbol}: {position['quantity']} shares @ ${position['entry_price']:.2f} | Current: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        # Statistics
        win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        print(f"\nTotal Trades: {self.total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print("-"*60)
    
    async def run(self):
        """主交易循環"""
        self.running = True
        logger.info("Starting automated trading...")
        
        while self.running:
            try:
                # Check market conditions
                if not await self.check_market_conditions():
                    logger.info("Market closed or conditions not suitable")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Generate signals
                signals = await self.generate_signals()
                
                # Execute trades based on signals
                for symbol, signal in signals.items():
                    if signal in ['BUY', 'SELL']:
                        await self.execute_trade(symbol, signal)
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Display status
                self.display_status()
                
                # Wait for next cycle
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping trading system...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
    
    async def shutdown(self):
        """關閉系統"""
        logger.info("Shutting down trading system...")
        
        # Close all positions
        for symbol in list(self.active_positions.keys()):
            logger.info(f"Closing position: {symbol}")
            await self.execute_trade(symbol, 'SELL')
        
        logger.info("All positions closed. System shutdown complete.")

async def main():
    """主程序"""
    print("""
    ╔════════════════════════════════════════════════════╗
    ║        LIVE AUTOMATED TRADING SYSTEM               ║
    ║              Capital.com Demo Account              ║
    ║                                                    ║
    ║  Risk Parameters:                                  ║
    ║  - Max Daily Loss: 2%                             ║
    ║  - Stop Loss: 5%                                  ║
    ║  - Take Profit: 10%                               ║
    ║  - Max Positions: 5                               ║
    ║                                                    ║
    ║         Press Ctrl+C to stop trading              ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    system = LiveTradingSystem()
    
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
    os.makedirs('logs', exist_ok=True)
    
    # Run the trading system
    asyncio.run(main())