"""
Test RL Trading Environment
測試強化學習交易環境
Cloud DE - Verification Script
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.rl_trading.trading_env import TradingEnvironment, Actions
from src.data.minute_data_pipeline import MinuteDataPipeline


def create_test_data(n_steps=1000):
    """創建測試數據"""
    dates = pd.date_range(start="2024-01-01", periods=n_steps, freq="5min")

    # 生成隨機遊走價格
    returns = np.random.normal(0.0001, 0.005, n_steps)
    prices = 100 * (1 + returns).cumprod()

    # 生成 OHLCV 數據
    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(prices[0])
    df["high"] = df[["open", "close"]].max(axis=1) * (
        1 + np.abs(np.random.normal(0, 0.002, n_steps))
    )
    df["low"] = df[["open", "close"]].min(axis=1) * (
        1 - np.abs(np.random.normal(0, 0.002, n_steps))
    )
    df["volume"] = np.random.randint(10000, 100000, n_steps)

    return df


def test_environment_basic():
    """測試環境基本功能"""
    print("\n" + "=" * 50)
    print("Testing Basic Environment Functions")
    print("=" * 50)

    # 創建測試數據
    create_test_data(1000)

    # 創建環境
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        commission=0.001,
        slippage=0.0005,
        window_size=20,
    )

    # 測試重置
    obs, info = env.reset()
    print("✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Initial portfolio value: ${info['portfolio_value']:.2f}")

    # 測試步進
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Step execution successful")
    print(f"  Action taken: {Actions(action).name}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Portfolio value: ${info['portfolio_value']:.2f}")

    return True


def test_trading_logic():
    """測試交易邏輯"""
    print("\n" + "=" * 50)
    print("Testing Trading Logic")
    print("=" * 50)

    create_test_data(500)
    env = TradingEnvironment(df=data, initial_balance=10000)

    obs, info = env.reset()
    initial_value = info["portfolio_value"]

    # 執行一系列交易
    actions_sequence = [
        Actions.BUY,  # 買入
        Actions.HOLD,  # 持有
        Actions.HOLD,  # 持有
        Actions.SELL,  # 賣出部分
        Actions.HOLD,  # 持有
        Actions.CLOSE,  # 平倉
    ]

    for i, action in enumerate(actions_sequence):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: {Actions(action).name}")
        print(f"  Position: {info['position']}")
        print(f"  Cash: ${info['cash']:.2f}")
        print(f"  Portfolio: ${info['portfolio_value']:.2f}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    final_value = info["portfolio_value"]
    total_return = (final_value - initial_value) / initial_value * 100
    print("\n✓ Trading logic test complete")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Total trades: {info['total_trades']}")
    print(f"  Win rate: {info['win_rate']:.2%}")

    return True


def test_random_agent():
    """測試隨機智能體"""
    print("\n" + "=" * 50)
    print("Testing Random Agent")
    print("=" * 50)

    create_test_data(2000)
    env = TradingEnvironment(df=data, initial_balance=10000)

    # 運行一個完整 episode
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0

    portfolio_values = [info["portfolio_value"]]

    while not done:
        # 隨機動作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        portfolio_values.append(info["portfolio_value"])
        step_count += 1

        done = terminated or truncated

        # 限制最大步數
        if step_count >= 1000:
            break

    print("✓ Random agent test complete")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final portfolio: ${info['portfolio_value']:.2f}")
    print(f"  Max drawdown: {info.get('max_drawdown', 0):.2%}")
    print(f"  Sharpe ratio: {info.get('sharpe_ratio', 0):.2f}")

    return portfolio_values


def test_minute_data_pipeline():
    """測試分鐘數據管道"""
    print("\n" + "=" * 50)
    print("Testing Minute Data Pipeline")
    print("=" * 50)

    pipeline = MinuteDataPipeline()

    # 測試下載
    ["AAPL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    print(f"Downloading {symbols[0]} data...")
    results = pipeline.download_data(
        symbols, start_date=start_date, end_date=end_date, interval="5min"
    )

    if results and symbols[0] in results:
        df = results[symbols[0]]
        print(f"✓ Downloaded {len(df)} records")

        # 驗證數據
        pipeline.validate_data(df)
        print(f"✓ Data quality score: {report['quality_score']:.2f}%")

        # 添加特徵
        df_with_features = pipeline.add_features(df)
        print(
            f"✓ Added {len(df_with_features.columns) - len(df.columns)} technical features"
        )

        return df_with_features
    else:
        print("✗ Failed to download data")
        return None


def test_gym_compatibility():
    """測試 Gym 相容性"""
    print("\n" + "=" * 50)
    print("Testing Gymnasium Compatibility")
    print("=" * 50)

    create_test_data(1000)
    env = TradingEnvironment(df=data)

    # 使用 gymnasium 的環境檢查器
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env)
        print("✓ Environment passes gymnasium compatibility check")
        return True
    except Exception as e:
        print(f"✗ Compatibility check failed: {e}")
        return False


def run_performance_test():
    """性能測試"""
    print("\n" + "=" * 50)
    print("Performance Test")
    print("=" * 50)

    import time

    create_test_data(5000)
    env = TradingEnvironment(df=data)

    # 測試步進速度
    obs, info = env.reset()

    start_time = time.time()
    n_steps = 1000

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start_time
    steps_per_second = n_steps / elapsed

    print("✓ Performance test complete")
    print(f"  Steps per second: {steps_per_second:.0f}")
    print(f"  Time per step: {elapsed/n_steps*1000:.2f} ms")

    # 檢查是否達到要求
    if steps_per_second > 1000:
        print("  ✓ Meets performance requirement (>1000 steps/s)")
    else:
        print("  ✗ Below performance requirement (>1000 steps/s)")

    return steps_per_second


def visualize_episode(portfolio_values):
    """視覺化交易結果"""
    print("\n" + "=" * 50)
    print("Visualizing Trading Episode")
    print("=" * 50)

    plt.figure(figsize=(12, 6))

    # 投資組合價值
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Value ($)")
    plt.grid(True, alpha=0.3)

    # 收益率
    plt.subplot(2, 1, 2)
    returns = pd.Series(portfolio_values).pct_change().fillna(0)
    plt.plot(returns)
    plt.title("Step Returns")
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存圖片
    plt.savefig("reports/rl_environment_test.png", dpi=100)
    print("✓ Visualization saved to reports/rl_environment_test.png")

    plt.show()


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("RL TRADING ENVIRONMENT TEST SUITE")
    print("Cloud DE - Task DT-001 & DT-003 Verification")
    print("=" * 60)

    results = {}

    # 1. 基本功能測試
    try:
        results["basic"] = test_environment_basic()
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        results["basic"] = False

    # 2. 交易邏輯測試
    try:
        results["trading"] = test_trading_logic()
    except Exception as e:
        print(f"✗ Trading logic test failed: {e}")
        results["trading"] = False

    # 3. Gym 相容性測試
    try:
        results["gym"] = test_gym_compatibility()
    except Exception as e:
        print(f"✗ Gym compatibility test failed: {e}")
        results["gym"] = False

    # 4. 隨機智能體測試
    try:
        portfolio_values = test_random_agent()
        results["random_agent"] = True
    except Exception as e:
        print(f"✗ Random agent test failed: {e}")
        results["random_agent"] = False
        portfolio_values = None

    # 5. 分鐘數據測試
    try:
        df = test_minute_data_pipeline()
        results["minute_data"] = df is not None
    except Exception as e:
        print(f"✗ Minute data test failed: {e}")
        results["minute_data"] = False

    # 6. 性能測試
    try:
        steps_per_sec = run_performance_test()
        results["performance"] = steps_per_sec > 1000
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        results["performance"] = False

    # 7. 視覺化
    if portfolio_values:
        try:
            visualize_episode(portfolio_values)
            results["visualization"] = True
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            results["visualization"] = False

    # 總結
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("RL Trading Environment is ready for PPO training!")
    else:
        print("\n⚠️ Some tests failed. Please review and fix.")

    return all_passed


if __name__ == "__main__":
    success = main()
