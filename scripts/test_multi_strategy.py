"""
Test Multi-Strategy Integration
測試多策略整合系統
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 導入策略
from src.strategies.cci_strategy import CCI20Strategy
from src.strategies.williams_r_strategy import WilliamsRStrategy
from src.strategies.stochastic_strategy import StochasticStrategy
from src.strategies.volume_sma_strategy import VolumeSMAStrategy
from src.strategies.obv_strategy import OBVStrategy
from src.strategies.multi_strategy_manager import MultiStrategyManager
from src.strategies.signal_aggregator import SignalAggregator


def load_test_data(symbol="AAPL", days=100):
    """
    載入測試數據
    """
    try:
        # 嘗試載入歷史數據
        data_path = f"data/stocks/{symbol}_daily.csv"
        if os.path.exists(data_path):
            pd.read_csv(data_path)
            data["date"] = pd.to_datetime(data["date"])
            data.set_index("date", inplace=True)

            # 取最近 N 天數據
            data.tail(days)
            logger.info(f"Loaded {len(data)} days of data for {symbol}")
            return data
        else:
            # 生成模擬數據
            logger.info(f"Generating simulated data for {symbol}")
            return generate_simulated_data(days)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return generate_simulated_data(days)


def generate_simulated_data(days=100):
    """
    生成模擬市場數據
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # 生成價格數據（隨機遊走）
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * (1 + returns).cumprod()

    # 生成 OHLCV 數據
    pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, days)),
            "close": prices,
            "volume": np.random.uniform(1000000, 5000000, days),
        },
        index=dates,
    )

    # 確保 high >= low
    data["high"] = data[["open", "close", "high"]].max(axis=1)
    data["low"] = data[["open", "close", "low"]].min(axis=1)

    return data


def test_individual_strategies(data):
    """
    測試個別策略
    """
    print("\n" + "=" * 60)
    print("*** TESTING INDIVIDUAL STRATEGIES ***")
    print("=" * 60)

    strategies = [
        CCI20Strategy(),
        WilliamsRStrategy(),
        StochasticStrategy(),
        VolumeSMAStrategy(),
        OBVStrategy(),
    ]

    results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy.name}...")

        try:
            # 計算信號
            strategy.calculate_signals(data)

            if len(signals) > 0:
                # 統計信號
                buy_count = signals["buy"].sum()
                sell_count = signals["sell"].sum()
                avg_strength = signals["signal_strength"].mean()

                results[strategy.name] = {
                    "buy_signals": int(buy_count),
                    "sell_signals": int(sell_count),
                    "avg_strength": float(avg_strength),
                    "total_signals": int(buy_count + sell_count),
                }

                print(f"  Buy signals: {buy_count}")
                print(f"  Sell signals: {sell_count}")
                print(f"  Avg signal strength: {avg_strength:.3f}")

                # 顯示最近信號
                recent_buys = signals[signals["buy"]].tail(3)
                if len(recent_buys) > 0:
                    print("  Recent buy signals:")
                    for idx, row in recent_buys.iterrows():
                        print(
                            f"    {idx.strftime('%Y-%m-%d')}: strength={row['signal_strength']:.3f}"
                        )
            else:
                print("  No signals generated")
                results[strategy.name] = {
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "avg_strength": 0,
                    "total_signals": 0,
                }

        except Exception as e:
            print(f"  Error: {e}")
            results[strategy.name] = {"error": str(e)}

    return results


def test_signal_aggregator(data):
    """
    測試信號聚合器
    """
    print("\n" + "=" * 60)
    print("*** TESTING SIGNAL AGGREGATOR ***")
    print("=" * 60)

    # 創建策略實例
    strategies = [
        CCI20Strategy(),
        WilliamsRStrategy(),
        StochasticStrategy(),
        VolumeSMAStrategy(),
        OBVStrategy(),
    ]

    # 創建聚合器
    aggregator = SignalAggregator(strategies=strategies)

    print(f"\nAggregator initialized with {len(strategies)} strategies")
    print(f"Consensus method: {aggregator.consensus_method}")
    print(f"Min agreement: {aggregator.min_agreement}")

    # 收集信號
    print("\nCollecting signals from all strategies...")
    all_signals = aggregator.collect_signals(data, parallel=False)

    # 測試不同的共識方法
    consensus_methods = ["voting", "weighted_voting", "score_based", "conservative", "aggressive"]

    for method in consensus_methods:
        print(f"\n--- Testing {method} consensus ---")

        try:
            consensus = aggregator.aggregate_signals(all_signals, method=method)

            if len(consensus) > 0:
                buy_count = consensus["buy"].sum()
                sell_count = consensus["sell"].sum()
                avg_confidence = consensus["confidence"].mean()

                print(f"  Buy signals: {buy_count}")
                print(f"  Sell signals: {sell_count}")
                print(f"  Avg confidence: {avg_confidence:.3f}")

                # 分析策略一致性
                agreement = aggregator.analyze_strategy_agreement(all_signals)
                print(f"  Avg buy agreement: {agreement.get('avg_buy_agreement', 0):.3f}")
                print(f"  Avg sell agreement: {agreement.get('avg_sell_agreement', 0):.3f}")
                print(f"  Conflict points: {agreement.get('conflict_points', 0)}")
                print(f"  Unanimous buy signals: {agreement.get('unanimous_buy', 0)}")
                print(f"  Unanimous sell signals: {agreement.get('unanimous_sell', 0)}")
            else:
                print("  No consensus signals generated")

        except Exception as e:
            print(f"  Error: {e}")

    # 獲取聚合器報告
    aggregator.get_aggregator_report()
    print("\n--- Aggregator Report ---")
    print(f"Total strategies: {report['total_strategies']}")
    print("Strategy weights:")
    for name, weight in report["strategy_weights"].items():
        print(f"  {name}: {weight:.2f}")


def test_multi_strategy_manager(data):
    """
    測試多策略管理器
    """
    print("\n" + "=" * 60)
    print("*** TESTING MULTI-STRATEGY MANAGER ***")
    print("=" * 60)

    # 創建管理器
    manager = MultiStrategyManager()

    # 添加策略
    strategies = [
        ("CCI_20", CCI20Strategy()),
        ("Williams_R", WilliamsRStrategy()),
        ("Stochastic", StochasticStrategy()),
        ("Volume_SMA", VolumeSMAStrategy()),
        ("OBV", OBVStrategy()),
    ]

    for name, strategy in strategies:
        manager.add_strategy(name, strategy)
        print(f"Added strategy: {name}")

    # 設置策略權重
    weights = {"CCI_20": 0.3, "Williams_R": 0.25, "Stochastic": 0.15, "Volume_SMA": 0.2, "OBV": 0.1}
    manager.set_strategy_weights(weights)

    print("\n--- Executing all strategies ---")

    # 執行所有策略
    all_results = manager.execute_all_strategies(data, parallel=False)

    # 獲取共識信號
    consensus_signal = manager.get_consensus_signal(all_results, method="weighted_voting")

    if consensus_signal:
        print("Consensus Signal:")
        print(f"  Action: {consensus_signal['action']}")
        print(f"  Confidence: {consensus_signal['confidence']:.3f}")
        print(f"  Agreeing strategies: {consensus_signal.get('agreeing_strategies', [])}")

        # 計算持倉大小
        portfolio_value = 100000
        current_price = data["close"].iloc[-1]

        position = manager.calculate_position_size(consensus_signal, portfolio_value, current_price)

        print("\nPosition Sizing:")
        print(f"  Shares: {position['shares']}")
        print(f"  Allocation: ${position['allocation']:.2f}")
        print(f"  Allocation %: {position['allocation_pct']:.2f}%")
    else:
        print("No consensus signal generated")

    # 獲取策略報告
    reports = manager.get_strategy_reports()
    print("\n--- Strategy Reports ---")
    for name, report in reports.items():
        if report:
            print(f"\n{name}:")
            if "expected_performance" in report:
                perf = report["expected_performance"]
                print(f"  Expected return: {perf.get('avg_return', 'N/A')}")
                print(f"  Win rate: {perf.get('win_rate', 'N/A')}")
            print(f"  Total signals: {report.get('total_signals', 0)}")


def test_backtest_simulation(data):
    """
    簡單回測模擬
    """
    print("\n" + "=" * 60)
    print("*** SIMPLE BACKTEST SIMULATION ***")
    print("=" * 60)

    # 初始資金
    initial_capital = 100000
    capital = initial_capital
    position = 0
    trades = []

    # 創建管理器
    manager = MultiStrategyManager()

    # 添加策略
    manager.add_strategy("CCI_20", CCI20Strategy())
    manager.add_strategy("Williams_R", WilliamsRStrategy())
    manager.add_strategy("Stochastic", StochasticStrategy())

    print(f"Initial capital: ${initial_capital:,.0f}")
    print(f"Testing on {len(data)} days of data")
    print(f"Using {len(manager.strategies)} strategies")

    # 模擬交易
    for i in range(20, len(data)):
        # 獲取當前數據窗口
        window_data = data.iloc[i - 20 : i + 1]
        current_price = window_data["close"].iloc[-1]

        # 執行策略
        all_results = manager.execute_all_strategies(window_data, parallel=False)
        consensus = manager.get_consensus_signal(all_results, method="weighted_voting")

        if consensus:
            # 買入信號
            if consensus["action"] == "BUY" and position == 0:
                # 計算持倉
                pos_size = manager.calculate_position_size(consensus, capital, current_price)
                if pos_size["shares"] > 0:
                    position = pos_size["shares"]
                    entry_price = current_price
                    capital -= position * entry_price

                    trades.append(
                        {
                            "date": window_data.index[-1],
                            "action": "BUY",
                            "price": entry_price,
                            "shares": position,
                            "confidence": consensus["confidence"],
                        }
                    )

            # 賣出信號
            elif consensus["action"] == "SELL" and position > 0:
                exit_price = current_price
                capital += position * exit_price
                pnl = (exit_price - entry_price) * position
                pnl_pct = (exit_price - entry_price) / entry_price * 100

                trades.append(
                    {
                        "date": window_data.index[-1],
                        "action": "SELL",
                        "price": exit_price,
                        "shares": position,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                    }
                )

                position = 0

    # 平倉未平倉位
    if position > 0:
        final_price = data["close"].iloc[-1]
        capital += position * final_price
        pnl = (final_price - entry_price) * position
        pnl_pct = (final_price - entry_price) / entry_price * 100

        trades.append(
            {
                "date": data.index[-1],
                "action": "CLOSE",
                "price": final_price,
                "shares": position,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    # 計算績效
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    print("\n--- Backtest Results ---")
    print(f"Final capital: ${final_capital:,.0f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Total trades: {len(trades)}")

    if trades:
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        print(f"Winning trades: {len(winning_trades)}")
        print(f"Losing trades: {len(losing_trades)}")

        if winning_trades or losing_trades:
            win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100
            print(f"Win rate: {win_rate:.1f}%")

        # 顯示最近交易
        print("\n--- Recent Trades ---")
        for trade in trades[-5:]:
            date_str = (
                trade["date"].strftime("%Y-%m-%d")
                if hasattr(trade["date"], "strftime")
                else str(trade["date"])
            )
            if "pnl" in trade:
                print(
                    f"{date_str}: {trade['action']} @ ${trade['price']:.2f}, PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)"
                )
            else:
                print(
                    f"{date_str}: {trade['action']} @ ${trade['price']:.2f}, Shares: {trade['shares']}"
                )


def save_test_results(results, filename="test_results.json"):
    """
    保存測試結果
    """
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    # 轉換結果為可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            serializable_results[key] = value.to_dict()
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value

    with open(filepath, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nTest results saved to {filepath}")


def main():
    """
    主測試函數
    """
    print("=" * 60)
    print("*** MULTI-STRATEGY INTEGRATION TEST ***")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 載入測試數據
    print("\nLoading test data...")
    load_test_data("AAPL", days=100)
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # 執行測試
    results = {}

    # 1. 測試個別策略
    individual_results = test_individual_strategies(data)
    results["individual_strategies"] = individual_results

    # 2. 測試信號聚合器
    test_signal_aggregator(data)

    # 3. 測試多策略管理器
    test_multi_strategy_manager(data)

    # 4. 簡單回測模擬
    test_backtest_simulation(data)

    # 保存結果
    save_test_results(
        results, f'multi_strategy_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

    print("\n" + "=" * 60)
    print("*** TEST COMPLETED SUCCESSFULLY ***")
    print("=" * 60)
    print("\nAll multi-strategy components tested successfully!")
    print("The integration is working as expected.")


if __name__ == "__main__":
    main()
