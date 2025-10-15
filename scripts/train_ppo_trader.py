"""
Train PPO Trading Agent
訓練 PPO 日內交易智能體
Cloud Quant - Training Script
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path

from src.rl_trading.trading_env import TradingEnvironment
from src.rl_trading.ppo_agent import PPOTrainer, PPOConfig
from src.data.minute_data_pipeline import MinuteData

# 設置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_training_data(symbol="AAPL", days=30):
    """
    準備訓練數據

    Args:
        symbol: 股票代碼
        days: 數據天數

    Returns:
        訓練數據 DataFrame
    """
    logger.info(f"Preparing training data for {symbol}...")

    # 嘗試下載真實數據
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = MinuteData.get(
            symbols=symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="5min",
        )

        if data is not None and len(data) > 100:
            logger.info(f"Loaded {len(data)} real data points")
            return data
    except Exception as e:
        logger.warning(f"Failed to load real data: {e}")

    # 生成模擬數據
    logger.info("Generating simulated data...")
    n_steps = days * 78  # 約每天 78 個 5 分鐘 bar (6.5 小時)
    dates = pd.date_range(start="2024-01-01", periods=n_steps, freq="5min")

    # 生成更真實的價格模擬
    trend = np.sin(np.linspace(0, 4 * np.pi, n_steps)) * 5  # 長期趨勢
    noise = np.random.normal(0, 0.5, n_steps)  # 短期噪音
    volatility = np.random.normal(0, 0.02, n_steps)  # 波動

    prices = 100 + trend + np.cumsum(noise) * 0.1
    prices = prices * (1 + volatility)

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(prices[0]) * (1 + np.random.normal(0, 0.001, n_steps))
    df["high"] = df[["open", "close"]].max(axis=1) * (
        1 + np.abs(np.random.normal(0, 0.002, n_steps))
    )
    df["low"] = df[["open", "close"]].min(axis=1) * (
        1 - np.abs(np.random.normal(0, 0.002, n_steps))
    )
    df["volume"] = np.random.randint(50000, 200000, n_steps)

    logger.info(f"Generated {len(df)} simulated data points")
    return df


def create_training_env(data):
    """
    創建訓練環境

    Args:
        data: 訓練數據

    Returns:
        交易環境
    """
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        commission=0.001,
        slippage=0.0005,
        max_position=100,
        window_size=20,
        reward_scaling=1e-4,
    )
    return env


def train_ppo_agent(env, config, total_timesteps=100000):
    """
    訓練 PPO 智能體

    Args:
        env: 交易環境
        config: PPO 配置
        total_timesteps: 總訓練步數

    Returns:
        訓練好的智能體
    """
    logger.info("Starting PPO training...")
    logger.info(f"Device: {config.device}")
    logger.info(f"Total timesteps: {total_timesteps}")

    # 創建訓練器
    trainer = PPOTrainer(env, config)

    # 訓練
    trainer.train(total_timesteps=total_timesteps, log_interval=10, save_interval=50)

    return trainer


def evaluate_agent(trainer, env, n_episodes=10):
    """
    評估智能體性能

    Args:
        trainer: PPO 訓練器
        env: 交易環境
        n_episodes: 評估回合數

    Returns:
        評估結果
    """
    logger.info(f"Evaluating agent over {n_episodes} episodes...")

    results = trainer.evaluate(n_episodes)

    logger.info("Evaluation Results:")
    logger.info(f"  Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    logger.info(f"  Mean episode length: {results['mean_length']:.0f}")

    if "mean_win_rate" in results:
        logger.info(f"  Mean win rate: {results['mean_win_rate']:.2%}")
    if "mean_max_drawdown" in results:
        logger.info(f"  Mean max drawdown: {results['mean_max_drawdown']:.2%}")

    return results


def backtest_strategy(trainer, test_data):
    """
    回測策略

    Args:
        trainer: 訓練好的 PPO 智能體
        test_data: 測試數據

    Returns:
        回測結果
    """
    logger.info("Running backtest...")

    # 創建測試環境
    test_env = TradingEnvironment(
        df=test_data, initial_balance=10000, commission=0.001, slippage=0.0005
    )

    # 運行回測
    obs, info = test_env.reset()
    done = False

    portfolio_values = [info["portfolio_value"]]
    positions = [info["position"]]
    actions_taken = []
    prices = []

    while not done:
        # 獲取動作
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=trainer.config.device)
            obs_tensor = obs_tensor.unsqueeze(0)
            action, _, _, _ = trainer.model.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]

        # 執行動作
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        # 記錄
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["position"])
        actions_taken.append(action)
        prices.append(info["current_price"])

    # 計算指標
    portfolio_values = np.array(portfolio_values)
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]

    total_return = (final_value - initial_value) / initial_value

    # 計算夏普比率
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    # 計算最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    backtest_results = {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
        "num_trades": info.get("total_trades", 0),
        "win_rate": info.get("win_rate", 0),
        "portfolio_values": portfolio_values.tolist(),
        "positions": positions,
        "actions": actions_taken,
        "prices": prices,
    }

    logger.info("Backtest Results:")
    logger.info(f"  Total return: {total_return:.2%}")
    logger.info(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    logger.info(f"  Max drawdown: {max_drawdown:.2%}")
    logger.info(f"  Final value: ${final_value:.2f}")
    logger.info(f"  Number of trades: {backtest_results['num_trades']}")
    logger.info(f"  Win rate: {backtest_results['win_rate']:.2%}")

    return backtest_results


def plot_results(trainer, backtest_results):
    """
    繪製訓練和回測結果

    Args:
        trainer: PPO 訓練器
        backtest_results: 回測結果
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # 1. 訓練獎勵曲線
    if trainer.training_history:
        history_df = pd.DataFrame(trainer.training_history)
        axes[0, 0].plot(history_df["timesteps"], history_df["mean_reward"], label="Mean", alpha=0.7)
        axes[0, 0].plot(history_df["timesteps"], history_df["best_reward"], label="Best", alpha=0.7)
        axes[0, 0].set_xlabel("Timesteps")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].set_title("Training Progress")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. 損失曲線
    if trainer.training_history:
        axes[0, 1].plot(
            history_df["iteration"], history_df["pg_loss"], label="Policy Loss", alpha=0.7
        )
        axes[0, 1].plot(
            history_df["iteration"], history_df["value_loss"], label="Value Loss", alpha=0.7
        )
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Training Losses")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 投資組合價值
    portfolio_values = backtest_results["portfolio_values"]
    axes[1, 0].plot(portfolio_values)
    axes[1, 0].set_xlabel("Time Steps")
    axes[1, 0].set_ylabel("Portfolio Value ($)")
    axes[1, 0].set_title("Portfolio Value During Backtest")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 持倉變化
    positions = backtest_results["positions"]
    axes[1, 1].plot(positions)
    axes[1, 1].set_xlabel("Time Steps")
    axes[1, 1].set_ylabel("Position")
    axes[1, 1].set_title("Position Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    # 5. 動作分佈
    actions = backtest_results["actions"]
    action_counts = pd.Series(actions).value_counts().sort_index()
    action_names = ["HOLD", "BUY", "SELL", "CLOSE"]
    axes[2, 0].bar([action_names[i] for i in action_counts.index], action_counts.values)
    axes[2, 0].set_xlabel("Action")
    axes[2, 0].set_ylabel("Count")
    axes[2, 0].set_title("Action Distribution")
    axes[2, 0].grid(True, alpha=0.3)

    # 6. 收益分佈
    returns = pd.Series(portfolio_values).pct_change().dropna()
    axes[2, 1].hist(returns, bins=50, alpha=0.7, edgecolor="black")
    axes[2, 1].set_xlabel("Return")
    axes[2, 1].set_ylabel("Frequency")
    axes[2, 1].set_title("Return Distribution")
    axes[2, 1].axvline(x=0, color="r", linestyle="--", alpha=0.5)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存圖片
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "ppo_training_results.png", dpi=100, bbox_inches="tight")
    logger.info(f"Results plot saved to {output_dir / 'ppo_training_results.png'}")

    plt.show()


def save_results(trainer, backtest_results, evaluation_results):
    """
    保存所有結果

    Args:
        trainer: PPO 訓練器
        backtest_results: 回測結果
        evaluation_results: 評估結果
    """
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    # 保存訓練歷史
    if trainer.training_history:
        history_df = pd.DataFrame(trainer.training_history)
        history_df.to_csv(output_dir / "ppo_training_history.csv", index=False)

    # 保存結果摘要
    summary = {
        "training": {
            "total_timesteps": trainer.total_timesteps,
            "n_episodes": trainer.n_episodes,
            "best_reward": float(trainer.best_reward),
        },
        "evaluation": evaluation_results,
        "backtest": {
            "total_return": backtest_results["total_return"],
            "sharpe_ratio": backtest_results["sharpe_ratio"],
            "max_drawdown": backtest_results["max_drawdown"],
            "num_trades": backtest_results["num_trades"],
            "win_rate": backtest_results["win_rate"],
        },
    }

    with open(output_dir / "ppo_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 保存模型
    model_path = output_dir / "ppo_trader_final.pt"
    trainer.save_model(str(model_path))

    logger.info(f"All results saved to {output_dir}")


def main():
    """
    主訓練函數
    """
    print("\n" + "=" * 60)
    print("PPO TRADING AGENT TRAINING")
    print("Cloud Quant - Task DT-002")
    print("=" * 60)

    # 配置
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        n_epochs=10,
        batch_size=64,
        n_steps=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 超參數（可調整）
    SYMBOL = "AAPL"
    TRAINING_DAYS = 30
    TOTAL_TIMESTEPS = 10000  # 大幅減少以快速測試

    try:
        # 1. 準備數據
        print("\n1. Preparing training data...")
        train_data = prepare_training_data(SYMBOL, TRAINING_DAYS)

        # 分割訓練和測試數據
        split_idx = int(len(train_data) * 0.8)
        training_data = train_data.iloc[:split_idx]
        test_data = train_data.iloc[split_idx:]

        print(f"   Training data: {len(training_data)} points")
        print(f"   Test data: {len(test_data)} points")

        # 2. 創建環境
        print("\n2. Creating training environment...")
        env = create_training_env(training_data)

        # 3. 訓練智能體
        print("\n3. Training PPO agent...")
        trainer = train_ppo_agent(env, config, TOTAL_TIMESTEPS)

        # 4. 評估智能體
        print("\n4. Evaluating agent...")
        evaluation_results = evaluate_agent(trainer, env, n_episodes=10)

        # 5. 回測策略
        print("\n5. Running backtest...")
        backtest_results = backtest_strategy(trainer, test_data)

        # 6. 繪製結果
        print("\n6. Plotting results...")
        plot_results(trainer, backtest_results)

        # 7. 保存結果
        print("\n7. Saving results...")
        save_results(trainer, backtest_results, evaluation_results)

        # 總結
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nFinal Performance:")
        print(f"  Training best reward: {trainer.best_reward:.4f}")
        print(f"  Backtest total return: {backtest_results['total_return']:.2%}")
        print(f"  Backtest Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Backtest max drawdown: {backtest_results['max_drawdown']:.2%}")

        # 檢查是否達到目標
        if backtest_results["sharpe_ratio"] > 1.5:
            print("\n✓ Achieved target Sharpe ratio (>1.5)!")
        else:
            print(
                f"\n✗ Below target Sharpe ratio (1.5), got {backtest_results['sharpe_ratio']:.2f}"
            )

        if backtest_results["total_return"] > 0.2:
            print("✓ Achieved target return (>20%)!")
        else:
            print(f"✗ Below target return (20%), got {backtest_results['total_return']:.2%}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
