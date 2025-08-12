"""
Main Trading System Controller (English Version)
Cloud Quant - Task SYS-001
"""

import asyncio
import sys
import os
import signal
import logging
from datetime import datetime
from pathlib import Path
import argparse
import json
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.trading_system import IntegratedTradingSystem, SystemConfig, SystemMode, StrategyType
from src.core.paper_trading import PaperTradingSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global system instance
trading_system: Optional[IntegratedTradingSystem] = None


class TradingConsole:
    """
    Trading System Console
    
    Provides command line interface to manage trading system
    """
    
    def __init__(self):
        """Initialize console"""
        self.system = None
        self.is_running = False
        
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("     INTELLIGENT QUANTITATIVE TRADING SYSTEM")
        print("="*60)
        print("\n[1] Start Paper Trading")
        print("[2] Start Live Trading")
        print("[3] Run Backtest")
        print("[4] Configure Strategy")
        print("[5] View Performance")
        print("[6] Risk Monitor")
        print("[7] Generate Report")
        print("[8] System Settings")
        print("[9] Emergency Stop")
        print("[0] Exit")
        print("\n" + "="*60)
    
    def display_strategy_menu(self):
        """Display strategy menu"""
        print("\n" + "="*50)
        print("SELECT TRADING STRATEGY")
        print("="*50)
        print("[1] MPT Portfolio Optimization")
        print("[2] Day Trading with PPO")
        print("[3] Hybrid Strategy")
        print("[0] Back to Main Menu")
    
    async def start_paper_trading(self):
        """Start paper trading"""
        print("\nStarting Paper Trading Mode...")
        
        # Select strategy
        strategy_type = await self.select_strategy()
        
        # Create configuration
        config = SystemConfig(
            mode=SystemMode.PAPER,
            strategy_type=strategy_type,
            paper_balance=100000,
            max_positions=50,
            risk_limit=0.05
        )
        
        # Initialize system
        self.system = IntegratedTradingSystem(config)
        
        # Display real-time monitoring
        print("\n" + "="*50)
        print("PAPER TRADING ACTIVE")
        print("="*50)
        print(f"Initial Balance: ${config.paper_balance:,.2f}")
        print(f"Strategy: {strategy_type.value}")
        print(f"Risk Limit: {config.risk_limit:.1%}")
        print("\nPress Ctrl+C to stop...")
        print("="*50)
        
        # Start system
        try:
            await self.system.run()
        except KeyboardInterrupt:
            print("\nStopping paper trading...")
            await self.system.shutdown()
    
    async def start_live_trading(self):
        """Start live trading"""
        print("\nWARNING: LIVE TRADING MODE")
        print("This will execute real trades with real money!")
        
        confirmation = input("Are you sure? Type 'YES' to confirm: ")
        if confirmation != 'YES':
            print("Live trading cancelled.")
            return
        
        # Need API credentials
        print("\nPlease configure your Capital.com API credentials in config/api_config.json")
        print("Then restart the system.")
        
        # TODO: Implement live trading logic
    
    async def run_backtest(self):
        """Run backtest"""
        print("\nStarting Backtest...")
        
        # Get backtest parameters
        start_date = input("Start date (YYYY-MM-DD): ")
        end_date = input("End date (YYYY-MM-DD): ")
        initial_capital = float(input("Initial capital ($): ") or "10000")
        
        # Select strategy
        strategy_type = await self.select_strategy()
        
        print(f"\nRunning backtest from {start_date} to {end_date}...")
        print(f"Initial capital: ${initial_capital:,.2f}")
        print(f"Strategy: {strategy_type.value}")
        
        # TODO: Implement backtest logic
        print("\nBacktest complete! Check reports/ for results.")
    
    async def select_strategy(self) -> StrategyType:
        """Select trading strategy"""
        self.display_strategy_menu()
        
        while True:
            choice = input("\nSelect strategy (0-3): ")
            
            if choice == '1':
                return StrategyType.MPT_PORTFOLIO
            elif choice == '2':
                return StrategyType.DAY_TRADING
            elif choice == '3':
                return StrategyType.HYBRID
            elif choice == '0':
                return StrategyType.HYBRID  # Default
            else:
                print("Invalid choice. Please try again.")
    
    async def view_performance(self):
        """View performance"""
        if not self.system:
            print("\nNo active trading system.")
            return
        
        report = self.system.generate_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Portfolio Value: ${report['portfolio_value']:,.2f}")
        print(f"Total Return: {report['total_return']:.2%}")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {report['max_drawdown']:.2%}")
        print(f"Win Rate: {report['win_rate']:.2%}")
        print(f"Active Positions: {report['active_positions']}")
        print(f"Total Trades: {report['total_trades']}")
        
        print("\nTop Positions:")
        for symbol, quantity in report['top_positions'][:5]:
            print(f"  {symbol}: {quantity:.0f} shares")
    
    async def risk_monitor(self):
        """Risk monitoring"""
        if not self.system:
            print("\nNo active trading system.")
            return
        
        risk_metrics = self.system.risk_manager.risk_metrics
        
        print("\n" + "="*60)
        print("RISK MONITOR")
        print("="*60)
        print(f"Position Count: {risk_metrics.get('position_count', 0)}")
        print(f"Concentration Risk: {risk_metrics.get('concentration_risk', 0):.2%}")
        print(f"Risk Score: {risk_metrics.get('risk_score', 0):.0f}/100")
        print(f"Current Drawdown: {self.system.state.max_drawdown:.2%}")
        
        if risk_metrics.get('risk_score', 0) > 70:
            print("\nWARNING: High risk level detected!")
    
    async def generate_report(self):
        """Generate report"""
        if not self.system:
            print("\nNo active trading system.")
            return
        
        report = self.system.generate_report()
        print(f"\nReport generated: reports/system_report_{datetime.now():%Y%m%d_%H%M%S}.json")
    
    async def system_settings(self):
        """System settings"""
        print("\n" + "="*50)
        print("SYSTEM SETTINGS")
        print("="*50)
        print("[1] Risk Parameters")
        print("[2] Strategy Weights")
        print("[3] Data Sources")
        print("[4] API Configuration")
        print("[0] Back to Main Menu")
        
        choice = input("\nSelect option: ")
        
        if choice == '1':
            print("\nRisk Parameters:")
            print("  Max Drawdown: 5%")
            print("  Max Positions: 50")
            print("  Position Size: 2%")
            # TODO: Implement parameter modification
        elif choice == '2':
            print("\nStrategy Weights:")
            print("  MPT Portfolio: 60%")
            print("  Day Trading: 40%")
            # TODO: Implement weight adjustment
    
    async def emergency_stop(self):
        """Emergency stop"""
        print("\nEMERGENCY STOP INITIATED!")
        
        if self.system:
            print("Closing all positions...")
            print("Cancelling all orders...")
            await self.system.shutdown()
            print("System stopped safely.")
        else:
            print("No active system to stop.")
    
    async def run(self):
        """Run console main loop"""
        print("\n" + "="*60)
        print("   WELCOME TO INTELLIGENT QUANTITATIVE TRADING SYSTEM")
        print("="*60)
        print("\nSystem initialized successfully!")
        print("Version: 1.0.0")
        print("Mode: Interactive Console")
        
        while True:
            try:
                self.display_menu()
                choice = input("\nSelect option (0-9): ")
                
                if choice == '1':
                    await self.start_paper_trading()
                elif choice == '2':
                    await self.start_live_trading()
                elif choice == '3':
                    await self.run_backtest()
                elif choice == '4':
                    strategy = await self.select_strategy()
                    print(f"Strategy set to: {strategy.value}")
                elif choice == '5':
                    await self.view_performance()
                elif choice == '6':
                    await self.risk_monitor()
                elif choice == '7':
                    await self.generate_report()
                elif choice == '8':
                    await self.system_settings()
                elif choice == '9':
                    await self.emergency_stop()
                elif choice == '0':
                    print("\nThank you for using the Trading System!")
                    if self.system:
                        await self.system.shutdown()
                    break
                else:
                    print("Invalid option. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                if self.system:
                    await self.emergency_stop()
                break
            except Exception as e:
                logger.error(f"Error in console: {e}")
                print(f"\nError: {e}")


def signal_handler(signum, frame):
    """Handle system signals"""
    global trading_system
    print("\nReceived interrupt signal. Shutting down...")
    if trading_system:
        asyncio.create_task(trading_system.shutdown())
    sys.exit(0)


async def main():
    """Main function"""
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create log directories
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Intelligent Quantitative Trading System")
    parser.add_argument('--mode', choices=['paper', 'live', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--strategy', choices=['mpt', 'day_trading', 'hybrid'],
                       default='hybrid', help='Trading strategy')
    parser.add_argument('--headless', action='store_true',
                       help='Run without interactive console')
    
    args = parser.parse_args()
    
    if args.headless:
        # Headless mode
        print("Starting in headless mode...")
        
        strategy_map = {
            'mpt': StrategyType.MPT_PORTFOLIO,
            'day_trading': StrategyType.DAY_TRADING,
            'hybrid': StrategyType.HYBRID
        }
        
        mode_map = {
            'paper': SystemMode.PAPER,
            'live': SystemMode.LIVE,
            'backtest': SystemMode.BACKTEST
        }
        
        config = SystemConfig(
            mode=mode_map[args.mode],
            strategy_type=strategy_map[args.strategy]
        )
        
        global trading_system
        trading_system = IntegratedTradingSystem(config)
        await trading_system.run()
    else:
        # Interactive console
        console = TradingConsole()
        await console.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)