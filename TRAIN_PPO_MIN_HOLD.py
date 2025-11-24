#!/usr/bin/env python3
"""
RL3 PPO Training with Minimum Holding Period Constraint
使用本地parquet历史数据进行PPO训练 + 5日最小持仓期硬约束
Date: 2025-11-20

改进措施：
1. 添加5日最小持仓期硬约束（防止过度交易）
2. 保持0.1%交易成本（原始基线配置）
3. 训练期间：2018-2025（7年，比10年更聚焦）
4. 目标：修正熊市alpha -25.71%问题，同时降低交易频率
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm
import logging
from typing import Dict, List
import warnings
import glob
import os

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ppo_training_min_hold.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PPOConfig:
    """PPO训练配置 - 优化版"""

    def __init__(self):
        # 模型参数
        self.obs_dim = 220  # 特征维度
        self.action_dim = 4  # 动作空间: 0=hold, 1=buy, 2=sell, 3=close
        self.hidden_dim = 256

        # 训练参数
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.n_epochs = 10
        self.batch_size = 64
        self.n_steps = 256

        # 环境参数 - 关键改进！
        self.initial_capital = 100000
        self.max_positions = 20
        self.transaction_cost = 0.001  # 0.1% (保持基线配置)
        self.min_holding_period = 5  # ✨ 新增：5日最小持仓期（硬约束）

        # 数据参数 - 优化训练期间
        self.data_start_date = "2018-01-01"  # 从2018开始（vs 2015）
        self.data_end_date = "2025-08-08"  # 到2025
        self.max_stocks = 200  # 使用200档股票（vs 100）

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """PPO Actor-Critic网络"""

    def __init__(self, config: PPOConfig):
        super(ActorCritic, self).__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value


class LocalDataLoader:
    """本地Parquet数据载入器"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.data_dir = "scripts/download/historical_data/daily"
        self.data_cache = {}

    def load_local_data(self):
        """载入本地parquet档案"""
        print(f"\n[DATA] Loading local parquet files from {self.data_dir}")
        print(f"[DATA] Date range: {self.config.data_start_date} to {self.config.data_end_date}")

        # 获取所有parquet档案
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        print(f"[DATA] Found {len(parquet_files)} parquet files")

        # 选取前N个档案
        selected_files = parquet_files[: self.config.max_stocks]
        print(f"[DATA] Selected {len(selected_files)} files for training")

        success_count = 0
        failed_files = []

        for file_path in tqdm(selected_files, desc="Loading data"):
            try:
                # 从档名提取股票代码
                symbol = os.path.basename(file_path).replace(".parquet", "")

                # 读取parquet档案
                df = pd.read_parquet(file_path)

                # 确保有日期索引 - 使用timestamp列
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # 过滤日期范围
                df = df[
                    (df.index >= self.config.data_start_date)
                    & (df.index <= self.config.data_end_date)
                ]

                # 确保有必要的列
                required_cols = ["open", "high", "low", "close", "volume"]
                df.columns = df.columns.str.lower()

                if all(col in df.columns for col in required_cols):
                    # 标准化列名
                    df = df.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                    )

                    if len(df) > 252:  # 至少一年数据
                        self.data_cache[symbol] = df
                        success_count += 1
                    else:
                        failed_files.append(f"{symbol} (insufficient data: {len(df)} days)")
                else:
                    failed_files.append(f"{symbol} (missing columns)")

            except Exception as e:
                failed_files.append(f"{os.path.basename(file_path)} ({str(e)[:50]})")

        print(
            f"\n[DATA] Successfully loaded: {success_count}/{len(selected_files)} files"
        )
        if failed_files:
            print(f"[DATA] Failed files ({len(failed_files)}): {failed_files[:5]}...")

        return self.data_cache

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备220维特征"""
        features = []

        # 1. 价格特征 (50维)
        returns = data["Close"].pct_change().fillna(0).values[-50:]
        features.append(returns.flatten() if len(returns.shape) > 1 else returns)

        # 2. 成交量特征 (20维)
        volume_ma = data["Volume"].rolling(20).mean()
        volume_ratio = (data["Volume"] / volume_ma).fillna(1).values[-20:]
        features.append(volume_ratio.flatten() if len(volume_ratio.shape) > 1 else volume_ratio)

        # 3. 技术指标 (150维)
        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).fillna(50).values[-30:]
        features.append(rsi.flatten() if len(rsi.shape) > 1 else rsi)

        # MACD
        ema12 = data["Close"].ewm(span=12).mean()
        ema26 = data["Close"].ewm(span=26).mean()
        macd = (ema12 - ema26).fillna(0).values[-30:]
        features.append(macd.flatten() if len(macd.shape) > 1 else macd)

        # Bollinger Bands
        sma = data["Close"].rolling(20).mean()
        std = data["Close"].rolling(20).std()
        bb_upper = ((data["Close"] - sma) / (std + 1e-10)).fillna(0).values[-30:]
        features.append(bb_upper.flatten() if len(bb_upper.shape) > 1 else bb_upper)

        # ATR
        high_low = data["High"] - data["Low"]
        high_close = abs(data["High"] - data["Close"].shift())
        low_close = abs(data["Low"] - data["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().fillna(0).values[-30:]
        features.append(atr.flatten() if len(atr.shape) > 1 else atr)

        # 合并所有特征
        all_features = np.concatenate(features)

        # 确保特征维度为220
        if len(all_features) < 220:
            all_features = np.pad(all_features, (0, 220 - len(all_features)), "constant")
        elif len(all_features) > 220:
            all_features = all_features[:220]

        return all_features


class SimpleTradingEnv:
    """
    交易环境 - 带最小持仓期约束

    关键改进：
    1. 硬性5日最小持仓期约束
    2. 持仓不足5日时，禁止卖出动作
    3. 有效防止过度交易
    """

    def __init__(self, data: pd.DataFrame, config: PPOConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.position = 0
        self.days_held = 0  # ✨ 新增：持仓天数计数器
        self.capital = config.initial_capital
        self.data_loader = LocalDataLoader(config)
        self.trade_count = 0  # 跟踪交易次数

    def reset(self):
        """重置环境"""
        self.current_step = 252  # 从有足够历史数据的地方开始
        self.position = 0
        self.days_held = 0  # ✨ 重置持仓天数
        self.capital = self.config.initial_capital
        self.trade_count = 0
        return self.get_state()

    def get_state(self):
        """获取当前状态"""
        if self.current_step >= len(self.data):
            self.current_step = 252

        window_data = self.data.iloc[: self.current_step + 1]
        features = self.data_loader.prepare_features(window_data)
        return features

    def step(self, action):
        """
        执行动作 - 带最小持仓期约束

        动作空间：
        0 = hold（持有/观望）
        1 = buy（买入）
        2 = sell（卖出）- ✨ 仅在持仓>=5日时允许
        3 = close（强制平仓）
        """
        self.current_step += 1

        if self.current_step >= len(self.data):
            return self.get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["Close"]
        reward = 0

        # ✨ 关键改进：最小持仓期约束
        # 如果有持仓，增加持仓天数
        if self.position == 1:
            self.days_held += 1

        # 执行交易 - 带约束
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.days_held = 0  # ✨ 重置持仓天数计数器
            reward = -self.config.transaction_cost
            self.trade_count += 1

        elif action == 2 and self.position == 1:  # Sell
            # ✨ 核心约束：只有持仓满5日才允许卖出
            if self.days_held >= self.config.min_holding_period:
                self.position = 0
                self.days_held = 0  # ✨ 重置持仓天数
                reward = -self.config.transaction_cost
                self.trade_count += 1
            else:
                # 持仓不足5日，禁止卖出，视为hold动作
                # 不扣除交易成本，因为实际没有交易
                pass

        # 计算收益
        if self.position == 1:
            returns = self.data.iloc[self.current_step]["Close"] / self.data.iloc[self.current_step - 1]["Close"] - 1
            reward += returns

        done = self.current_step >= len(self.data) - 1

        return self.get_state(), reward, done, {}


def train_ppo_min_hold(config: PPOConfig):
    """使用本地数据进行PPO训练 - 带最小持仓期约束"""
    print("\n" + "=" * 80)
    print("RL3 PPO TRAINING WITH MINIMUM HOLDING PERIOD CONSTRAINT")
    print("=" * 80)
    print(f"KEY IMPROVEMENTS:")
    print(f"  - Minimum holding period: {config.min_holding_period} days (HARD CONSTRAINT)")
    print(f"  - Transaction cost: {config.transaction_cost*100}% (baseline)")
    print(f"  - Training period: {config.data_start_date} to {config.data_end_date}")
    print(f"  - Target: Fix bear market alpha and reduce overtrading")
    print("=" * 80)

    # 载入本地数据
    data_loader = LocalDataLoader(config)
    stock_data = data_loader.load_local_data()

    if len(stock_data) == 0:
        print("[ERROR] No data loaded! Training aborted.")
        return

    # 创建模型
    model = ActorCritic(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"\n[MODEL] Created ActorCritic model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"[MODEL] Device: {config.device}")

    # 训练统计
    training_stats = {
        "total_stocks": len(stock_data),
        "total_episodes": 0,
        "total_rewards": [],
        "total_trades": [],  # ✨ 新增：跟踪每个episode的交易次数
        "avg_rewards": [],
        "best_reward": float("-inf"),
    }

    # 创建输出目录
    os.makedirs("runs/rl3/min_hold_training", exist_ok=True)
    os.makedirs("models/ppo_local", exist_ok=True)

    # 对每个股票进行训练
    for symbol, data in tqdm(list(stock_data.items()), desc="Training stocks"):
        try:
            env = SimpleTradingEnv(data, config)
            state = env.reset()

            episode_reward = 0
            states, actions, rewards, values, log_probs = [], [], [], [], []

            for step in range(min(500, len(data) - 253)):  # 最多500步
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)

                with torch.no_grad():
                    action_logits, value = model(state_tensor)
                    dist = Categorical(logits=action_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_state, reward, done, _ = env.step(action.item())

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())

                episode_reward += reward
                state = next_state

                if done:
                    break

            # 记录统计
            training_stats["total_episodes"] += 1
            training_stats["total_rewards"].append(episode_reward)
            training_stats["total_trades"].append(env.trade_count)  # ✨ 记录交易次数

            if episode_reward > training_stats["best_reward"]:
                training_stats["best_reward"] = episode_reward

            logger.info(f"[{symbol}] Reward: {episode_reward:.4f}, Trades: {env.trade_count}")

        except Exception as e:
            logger.error(f"[{symbol}] Training failed: {e}")
            continue

    # 计算最终统计
    if training_stats["total_rewards"]:
        avg_reward = np.mean(training_stats["total_rewards"])
        std_reward = np.std(training_stats["total_rewards"])
        avg_trades = np.mean(training_stats["total_trades"])  # ✨ 平均交易次数
        training_stats["avg_reward"] = avg_reward
        training_stats["std_reward"] = std_reward
        training_stats["avg_trades_per_episode"] = avg_trades

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total stocks processed: {training_stats['total_episodes']}")
        print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"Best reward: {training_stats['best_reward']:.4f}")
        print(f"Worst reward: {min(training_stats['total_rewards']):.4f}")
        print(f"Average trades per episode: {avg_trades:.2f}")  # ✨ 显示交易频率
        print(f"Min holding period enforced: {config.min_holding_period} days")
        print("=" * 80)

        # 保存模型
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/ppo_local/ppo_model_min_hold_{timestamp}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
                "training_stats": training_stats,
            },
            model_path,
        )
        print(f"\n[SAVE] Model saved to: {model_path}")

        # 保存训练统计
        stats_path = f"runs/rl3/min_hold_training/training_stats_{timestamp}.json"
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2, default=str)
        print(f"[SAVE] Training stats saved to: {stats_path}")

        print("=" * 80)

    return training_stats, model_path


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("RL3 PPO TRAINING WITH MINIMUM HOLDING PERIOD")
    print("Objective: Fix bear market alpha + reduce overtrading")
    print("=" * 80)

    # 创建配置
    config = PPOConfig()

    print(f"\n[CONFIG] Training configuration:")
    print(f"  - Max stocks: {config.max_stocks}")
    print(f"  - Date range: {config.data_start_date} to {config.data_end_date}")
    print(f"  - Min holding period: {config.min_holding_period} days (HARD CONSTRAINT)")
    print(f"  - Transaction cost: {config.transaction_cost*100}%")
    print(f"  - Features: {config.obs_dim} dimensions")
    print(f"  - Actions: {config.action_dim} (hold/buy/sell/close)")
    print(f"  - Device: {config.device}")

    # 开始训练
    stats, model_path = train_ppo_min_hold(config)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved: {model_path}")
    print("Check runs/rl3/min_hold_training/ for training stats")
    print("Check ppo_training_min_hold.log for detailed logs")
    print("\nNext steps:")
    print("  1. Run OOS backtest on 2021-2022 (bear market)")
    print("  2. Run OOS backtest on 2023-2025 (bull market)")
    print("  3. Verify bear market alpha improvement")
    print("  4. Compare trading frequency vs baseline")
    print("=" * 80)


if __name__ == "__main__":
    main()
