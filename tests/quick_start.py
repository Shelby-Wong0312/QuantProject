"""
Quick Start Trading System
快速啟動交易系統
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.paper_trading import PaperTradingSimulator
import pandas as pd
import numpy as np
from datetime import datetime


async def run_paper_trading_demo():
    """運行模擬交易演示"""
    print("\n" + "=" * 60)
    print("   INTELLIGENT QUANTITATIVE TRADING SYSTEM")
    print("   Paper Trading Demo")
    print("=" * 60)

    # 初始化模擬器
    simulator = PaperTradingSimulator(
        initial_balance=100000, commission_rate=0.001, slippage_rate=0.0005
    )

    print(f"\n💰 Initial Balance: ${simulator.account.initial_balance:,.2f}")
    print(f"📊 Commission Rate: {simulator.commission_rate:.2%}")
    print(f"📈 Slippage Rate: {simulator.slippage_rate:.2%}")

    # 模擬市場數據
    stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    prices = {
        "AAPL": 180 + np.random.normal(0, 5),
        "GOOGL": 140 + np.random.normal(0, 3),
        "MSFT": 380 + np.random.normal(0, 8),
        "AMZN": 170 + np.random.normal(0, 4),
        "TSLA": 250 + np.random.normal(0, 10),
    }

    simulator.update_market_prices(prices)

    print("\n🚀 Starting Paper Trading...\n")
    print("-" * 60)

    # 執行一些交易
    trades = [
        ("AAPL", "BUY", 100),
        ("GOOGL", "BUY", 50),
        ("MSFT", "BUY", 30),
        ("TSLA", "BUY", 20),
    ]

    for symbol, side, quantity in trades:
        await simulator.place_order(
            symbol=symbol, side=side, quantity=quantity, order_type="MARKET"
        )
        print(f"📝 Order placed: {side} {quantity} {symbol}")

    print("\n⏳ Simulating market movements...")
    await asyncio.sleep(2)

    # 更新價格（模擬市場變動）
    for i in range(5):
        print(f"\n🔄 Market Update {i+1}/5")

        # 隨機價格變動
        new_prices = {}
        for stock in stocks:
            change = np.random.normal(0, 0.02)  # 2% 波動
            new_prices[stock] = prices[stock] * (1 + change)

            direction = "📈" if change > 0 else "📉"
            print(f"  {stock}: ${new_prices[stock]:.2f} {direction} ({change:.2%})")

        simulator.update_market_prices(new_prices)
        prices = new_prices

        # 計算當前價值
        portfolio_value = simulator.calculate_portfolio_value()
        pnl = portfolio_value - simulator.account.initial_balance
        pnl_pct = pnl / simulator.account.initial_balance

        print(f"\n💼 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"💵 P&L: ${pnl:,.2f} ({pnl_pct:+.2%})")

        await asyncio.sleep(1)

    # 執行一些賣出
    print("\n📉 Executing some sells...")

    sell_trades = [
        ("AAPL", "SELL", 50),
        ("GOOGL", "SELL", 25),
    ]

    for symbol, side, quantity in sell_trades:
        await simulator.place_order(
            symbol=symbol, side=side, quantity=quantity, order_type="MARKET"
        )
        print(f"📝 Order placed: {side} {quantity} {symbol}")

    # 最終報告
    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE REPORT")
    print("=" * 60)

    metrics = simulator.get_performance_metrics()

    print("\n📊 Portfolio Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Portfolio Value: ${metrics['portfolio_value']:,.2f}")
    print(f"  Cash Balance: ${metrics['cash_balance']:,.2f}")
    print(f"  Positions Count: {metrics['positions_count']}")

    print("\n💰 P&L Summary:")
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Realized P&L: ${metrics['realized_pnl']:,.2f}")
    print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}")

    print("\n📈 Trading Statistics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Commission Paid: ${metrics['total_commission']:,.2f}")

    if metrics["positions_count"] > 0:
        print("\n🏦 Current Positions:")
        for symbol, position in simulator.positions.items():
            print(f"  {symbol}: {position.quantity:.0f} shares @ ${position.avg_price:.2f}")
            print(f"    Market Value: ${position.market_value:,.2f}")
            print(f"    Unrealized P&L: ${position.unrealized_pnl:,.2f}")

    # 保存狀態
    simulator.save_state("reports/paper_trading_state.json")
    print("\n💾 State saved to reports/paper_trading_state.json")

    print("\n✅ Paper Trading Demo Complete!")

    return simulator


async def run_signal_generation_demo():
    """運行信號生成演示"""
    print("\n" + "=" * 60)
    print("   SIGNAL GENERATION DEMO")
    print("=" * 60)

    from src.signals.signal_generator import SignalGenerator

    # 初始化信號生成器
    generator = SignalGenerator()

    # 生成測試數據
    dates = pd.date_range(start="2024-01-01", periods=200, freq="5min")

    # 模擬不同市場情況
    scenarios = {
        "AAPL": {"trend": 0.3, "volatility": 0.1},  # 上升趨勢
        "GOOGL": {"trend": -0.2, "volatility": 0.15},  # 下降趨勢
        "MSFT": {"trend": 0, "volatility": 0.05},  # 橫盤
    }

    print("\n📊 Generating Trading Signals...\n")

    for symbol, params in scenarios.items():
        # 生成數據
        trend = np.linspace(0, params["trend"] * 200, 200)
        noise = np.random.normal(0, params["volatility"], 200).cumsum()
        prices = 100 + trend + noise

        pd.DataFrame(
            {
                "open": prices + np.random.normal(0, 0.1, 200),
                "high": prices + abs(np.random.normal(0, 0.2, 200)),
                "low": prices - abs(np.random.normal(0, 0.2, 200)),
                "close": prices,
                "volume": np.random.randint(50000, 200000, 200),
            },
            index=dates,
        )

        # 生成信號
        signal = generator.generate_signal(data, symbol)

        # 顯示信號
        if signal.action == "BUY":
            emoji = "🟢"
        elif signal.action == "SELL":
            emoji = "🔴"
        else:
            emoji = "⚪"

        print(f"{emoji} {symbol}:")
        print(f"  Action: {signal.action}")
        print(f"  Strength: {signal.strength:.0f}/100")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Price: ${signal.price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Risk Score: {signal.risk_score:.0f}/100")
        print()

    print("✅ Signal Generation Complete!")


async def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("   🚀 INTELLIGENT QUANTITATIVE TRADING SYSTEM - QUICK START")
    print("=" * 70)

    while True:
        print("\n[1] Run Paper Trading Demo")
        print("[2] Run Signal Generation Demo")
        print("[3] Run Both Demos")
        print("[0] Exit")

        choice = input("\nSelect option: ")

        if choice == "1":
            await run_paper_trading_demo()
        elif choice == "2":
            await run_signal_generation_demo()
        elif choice == "3":
            await run_paper_trading_demo()
            print("\n" + "-" * 60 + "\n")
            await run_signal_generation_demo()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice!")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
