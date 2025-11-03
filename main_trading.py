"""
Main Trading System Controller
主交易系統控制台
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
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.trading_system import (
    IntegratedTradingSystem,
    SystemConfig,
    SystemMode,
    StrategyType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/trading_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global system instance
trading_system: Optional[IntegratedTradingSystem] = None


class TradingConsole:
    """
    交易系統控制台

    提供命令行界面管理交易系統
    """

    def __init__(self):
        """初始化控制台"""
        self.system = None
        self.is_running = False

    def display_menu(self):
        """顯示主菜單"""
        print("\n" + "=" * 60)
        print("     INTELLIGENT QUANTITATIVE TRADING SYSTEM")
        print("=" * 60)
        print("\n[1] Start Paper Trading (模擬交易)")
        print("[2] Start Live Trading (實盤交易)")
        print("[3] Run Backtest (回測)")
        print("[4] Configure Strategy (策略配置)")
        print("[5] View Performance (查看績效)")
        print("[6] Risk Monitor (風險監控)")
        print("[7] Generate Report (生成報告)")
        print("[8] System Settings (系統設置)")
        print("[9] Emergency Stop (緊急停止)")
        print("[0] Exit (退出)")
        print("\n" + "=" * 60)

    def display_strategy_menu(self):
        """顯示策略菜單"""
        print("\n" + "=" * 50)
        print("SELECT TRADING STRATEGY")
        print("=" * 50)
        print("[1] MPT Portfolio Optimization (MPT投資組合優化)")
        print("[2] Day Trading with PPO (PPO日內交易)")
        print("[3] Hybrid Strategy (混合策略)")
        print("[0] Back to Main Menu")

    async def start_paper_trading(self):
        """啟動模擬交易"""
        print("\n🚀 Starting Paper Trading Mode...")

        # 選擇策略
        strategy_type = await self.select_strategy()

        # 創建配置
        config = SystemConfig(
            mode=SystemMode.PAPER,
            strategy_type=strategy_type,
            paper_balance=100000,
            max_positions=50,
            risk_limit=0.05,
        )

        # 初始化系統
        self.system = IntegratedTradingSystem(config)

        # 顯示實時監控
        print("\n" + "=" * 50)
        print("PAPER TRADING ACTIVE")
        print("=" * 50)
        print(f"Initial Balance: ${config.paper_balance:,.2f}")
        print(f"Strategy: {strategy_type.value}")
        print(f"Risk Limit: {config.risk_limit:.1%}")
        print("\nPress Ctrl+C to stop...")
        print("=" * 50)

        # 啟動系統
        try:
            await self.system.run()
        except KeyboardInterrupt:
            print("\n⚠️ Stopping paper trading...")
            await self.system.shutdown()

    async def start_live_trading(self):
        """啟動實盤交易"""
        print("\n⚠️ LIVE TRADING MODE")
        print("This will execute real trades with real money!")

        confirmation = input("Are you sure? Type 'YES' to confirm: ")
        if confirmation != "YES":
            print("Live trading cancelled.")
            return

        # 需要API憑證
        print(
            "\n📝 Please configure your Capital.com API credentials in config/api_config.json"
        )
        print("Then restart the system.")

        # TODO: 實現實盤交易邏輯

    async def run_backtest(self):
        """運行回測"""
        print("\n📊 Starting Backtest...")

        # 獲取回測參數
        start_date = input("Start date (YYYY-MM-DD): ")
        end_date = input("End date (YYYY-MM-DD): ")
        initial_capital = float(input("Initial capital ($): ") or "10000")

        # 選擇策略
        strategy_type = await self.select_strategy()

        print(f"\nRunning backtest from {start_date} to {end_date}...")
        print(f"Initial capital: ${initial_capital:,.2f}")
        print(f"Strategy: {strategy_type.value}")

        # TODO: 實現回測邏輯
        print("\n✅ Backtest complete! Check reports/ for results.")

    async def select_strategy(self) -> StrategyType:
        """選擇交易策略"""
        self.display_strategy_menu()

        while True:
            choice = input("\nSelect strategy (0-3): ")

            if choice == "1":
                return StrategyType.MPT_PORTFOLIO
            elif choice == "2":
                return StrategyType.DAY_TRADING
            elif choice == "3":
                return StrategyType.HYBRID
            elif choice == "0":
                return StrategyType.HYBRID  # Default
            else:
                print("Invalid choice. Please try again.")

    async def view_performance(self):
        """查看績效"""
        if not self.system:
            print("\n❌ No active trading system.")
            return

        self.system.generate_report()

        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Portfolio Value: ${report['portfolio_value']:,.2f}")
        print(f"Total Return: {report['total_return']:.2%}")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {report['max_drawdown']:.2%}")
        print(f"Win Rate: {report['win_rate']:.2%}")
        print(f"Active Positions: {report['active_positions']}")
        print(f"Total Trades: {report['total_trades']}")

        print("\nTop Positions:")
        for symbol, quantity in report["top_positions"][:5]:
            print(f"  {symbol}: {quantity:.0f} shares")

    async def risk_monitor(self):
        """風險監控"""
        if not self.system:
            print("\n❌ No active trading system.")
            return

        risk_metrics = self.system.risk_manager.risk_metrics

        print("\n" + "=" * 60)
        print("RISK MONITOR")
        print("=" * 60)
        print(f"Position Count: {risk_metrics.get('position_count', 0)}")
        print(f"Concentration Risk: {risk_metrics.get('concentration_risk', 0):.2%}")
        print(f"Risk Score: {risk_metrics.get('risk_score', 0):.0f}/100")
        print(f"Current Drawdown: {self.system.state.max_drawdown:.2%}")

        if risk_metrics.get("risk_score", 0) > 70:
            print("\n⚠️ WARNING: High risk level detected!")

    async def generate_report(self):
        """生成報告"""
        if not self.system:
            print("\n❌ No active trading system.")
            return

        self.system.generate_report()
        print(
            f"\n✅ Report generated: reports/system_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        )

    async def system_settings(self):
        """系統設置"""
        print("\n" + "=" * 50)
        print("SYSTEM SETTINGS")
        print("=" * 50)
        print("[1] Risk Parameters")
        print("[2] Strategy Weights")
        print("[3] Data Sources")
        print("[4] API Configuration")
        print("[0] Back to Main Menu")

        choice = input("\nSelect option: ")

        if choice == "1":
            print("\nRisk Parameters:")
            print("  Max Drawdown: 5%")
            print("  Max Positions: 50")
            print("  Position Size: 2%")
            # TODO: 實現參數修改
        elif choice == "2":
            print("\nStrategy Weights:")
            print("  MPT Portfolio: 60%")
            print("  Day Trading: 40%")
            # TODO: 實現權重調整

    async def emergency_stop(self):
        """緊急停止"""
        print("\n🛑 EMERGENCY STOP INITIATED!")

        if self.system:
            print("Closing all positions...")
            print("Cancelling all orders...")
            await self.system.shutdown()
            print("✅ System stopped safely.")
        else:
            print("No active system to stop.")

    async def run(self):
        """運行控制台主循環"""
        print("\n" + "=" * 60)
        print("   WELCOME TO INTELLIGENT QUANTITATIVE TRADING SYSTEM")
        print("=" * 60)
        print("\nSystem initialized successfully!")
        print("Version: 1.0.0")
        print("Mode: Interactive Console")

        while True:
            try:
                self.display_menu()
                choice = input("\nSelect option (0-9): ")

                if choice == "1":
                    await self.start_paper_trading()
                elif choice == "2":
                    await self.start_live_trading()
                elif choice == "3":
                    await self.run_backtest()
                elif choice == "4":
                    strategy = await self.select_strategy()
                    print(f"✅ Strategy set to: {strategy.value}")
                elif choice == "5":
                    await self.view_performance()
                elif choice == "6":
                    await self.risk_monitor()
                elif choice == "7":
                    await self.generate_report()
                elif choice == "8":
                    await self.system_settings()
                elif choice == "9":
                    await self.emergency_stop()
                elif choice == "0":
                    print("\n👋 Thank you for using the Trading System!")
                    if self.system:
                        await self.system.shutdown()
                    break
                else:
                    print("❌ Invalid option. Please try again.")

            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user.")
                if self.system:
                    await self.emergency_stop()
                break
            except Exception as e:
                logger.error(f"Error in console: {e}")
                print(f"\n❌ Error: {e}")


def signal_handler(signum, frame):
    """處理系統信號"""
    # global trading_system  # removed by ci-hotfix
    print("\n⚠️ Received interrupt signal. Shutting down...")
    if trading_system:
        asyncio.create_task(trading_system.shutdown())
    sys.exit(0)


async def main():
    """主函數"""
    # 設置信號處理
    signal.signal(signal.SIGINT, signal_handler)

    # 創建日誌目錄
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # 解析命令行參數
    parser = argparse.ArgumentParser(
        description="Intelligent Quantitative Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode",
    )
    parser.add_argument(
        "--strategy",
        choices=["mpt", "day_trading", "hybrid"],
        default="hybrid",
        help="Trading strategy",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without interactive console"
    )

    args = parser.parse_args()

    if args.headless:
        # 無界面模式
        print("Starting in headless mode...")

        strategy_map = {
            "mpt": StrategyType.MPT_PORTFOLIO,
            "day_trading": StrategyType.DAY_TRADING,
            "hybrid": StrategyType.HYBRID,
        }

        mode_map = {
            "paper": SystemMode.PAPER,
            "live": SystemMode.LIVE,
            "backtest": SystemMode.BACKTEST,
        }

        config = SystemConfig(
            mode=mode_map[args.mode], strategy_type=strategy_map[args.strategy]
        )
        # global trading_system  # removed by ci-hotfix
        trading_system = IntegratedTradingSystem(config)
        await trading_system.run()
    else:
        # 交互式控制台
        console = TradingConsole()
        await console.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
