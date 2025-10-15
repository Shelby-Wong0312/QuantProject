"""
Run Trading System Demo
運行交易系統演示
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.paper_trading import PaperTradingSimulator
import numpy as np


async def run_paper_trading():
    """運行模擬交易"""
    print("\n" + "=" * 60)
    print("   PAPER TRADING SYSTEM DEMO")
    print("=" * 60)

    # 初始化模擬器
    simulator = PaperTradingSimulator(
        initial_balance=100000, commission_rate=0.001, slippage_rate=0.0005
    )

    print(f"\nInitial Balance: ${simulator.account.initial_balance:,.2f}")
    print(f"Commission Rate: {simulator.commission_rate:.2%}")
    print(f"Slippage Rate: {simulator.slippage_rate:.2%}")

    # 模擬市場數據
    stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    prices = {"AAPL": 180, "GOOGL": 140, "MSFT": 380, "AMZN": 170, "TSLA": 250}

    simulator.update_market_prices(prices)

    print("\n>>> Starting Paper Trading...")
    print("-" * 60)

    # 執行買入交易
    print("\n=== PLACING BUY ORDERS ===")
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
        if order_id:
            print(f"[ORDER] {side} {quantity} shares of {symbol}")

    # 顯示初始持倉
    portfolio_value = simulator.calculate_portfolio_value()
    print(f"\nInitial Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Cash Balance: ${simulator.account.cash_balance:,.2f}")
    print(f"Positions: {len(simulator.positions)}")

    # 模擬5個市場週期
    print("\n=== SIMULATING MARKET MOVEMENTS ===")

    for i in range(5):
        print(f"\n--- Market Update {i+1}/5 ---")

        # 隨機價格變動
        new_prices = {}
        for stock in stocks:
            change = np.random.uniform(-0.03, 0.03)  # ±3% 變動
            new_prices[stock] = prices[stock] * (1 + change)

            direction = "UP" if change > 0 else "DOWN"
            print(
                f"  {stock}: ${new_prices[stock]:.2f} [{direction} {abs(change):.2%}]"
            )

        simulator.update_market_prices(new_prices)
        prices = new_prices

        # 計算當前價值
        portfolio_value = simulator.calculate_portfolio_value()
        pnl = portfolio_value - simulator.account.initial_balance
        pnl_pct = pnl / simulator.account.initial_balance

        print("\nPortfolio Update:")
        print(f"  Value: ${portfolio_value:,.2f}")
        print(f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2%})")

        await asyncio.sleep(0.5)

    # 執行部分賣出
    print("\n=== PLACING SELL ORDERS ===")

    sell_trades = [
        ("AAPL", "SELL", 50),
        ("GOOGL", "SELL", 25),
    ]

    for symbol, side, quantity in sell_trades:
        await simulator.place_order(
            symbol=symbol, side=side, quantity=quantity, order_type="MARKET"
        )
        if order_id:
            print(f"[ORDER] {side} {quantity} shares of {symbol}")

    # 最終報告
    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE REPORT")
    print("=" * 60)

    metrics = simulator.get_performance_metrics()

    print("\nPORTFOLIO METRICS:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Portfolio Value: ${metrics['portfolio_value']:,.2f}")
    print(f"  Cash Balance: ${metrics['cash_balance']:,.2f}")
    print(f"  Positions Count: {metrics['positions_count']}")

    print("\nP&L SUMMARY:")
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Realized P&L: ${metrics['realized_pnl']:,.2f}")
    print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}")

    print("\nTRADING STATISTICS:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Commission Paid: ${metrics['total_commission']:,.2f}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

    if metrics["positions_count"] > 0:
        print("\nCURRENT POSITIONS:")
        for symbol, position in simulator.positions.items():
            print(f"  {symbol}:")
            print(f"    Quantity: {position.quantity:.0f} shares")
            print(f"    Avg Price: ${position.avg_price:.2f}")
            print(f"    Current Price: ${position.current_price:.2f}")
            print(f"    Market Value: ${position.market_value:,.2f}")
            print(f"    Unrealized P&L: ${position.unrealized_pnl:+,.2f}")

    # 保存狀態
    simulator.save_state("reports/paper_trading_demo.json")
    print("\n>>> State saved to reports/paper_trading_demo.json")

    print("\n" + "=" * 60)
    print("PAPER TRADING DEMO COMPLETE!")
    print("=" * 60)

    return simulator


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   INTELLIGENT QUANTITATIVE TRADING SYSTEM")
    print("   Paper Trading Demo")
    print("=" * 70)

    try:
        # 運行演示
        simulator = asyncio.run(run_paper_trading())

        print("\nDemo completed successfully!")
        print("Check reports/paper_trading_demo.json for detailed results.")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()
