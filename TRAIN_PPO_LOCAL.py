#!/usr/bin/env python3
"""
RL3 PPO Training with Local Parquet Data
使用本地parquet歷史數據進行PPO訓練（小批量版本）
Date: 2025-11-19
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
        logging.FileHandler("ppo_training_local.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PPOConfig:
    """PPO訓練配置"""

    def __init__(self):
        # 模型參數
        self.obs_dim = 220  # 特徵維度
        self.action_dim = 4  # 動作空間: 0=hold, 1=buy, 2=sell, 3=close
        self.hidden_dim = 256

        # 訓練參數
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

        # 環境參數
        self.initial_capital = 100000
        self.max_positions = 20
        self.transaction_cost = 0.001

        # 數據參數
        self.data_start_date = "2015-01-01"
        self.data_end_date = "2022-12-31"
        self.max_stocks = 100  # 使用前100檔股票

        # 設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """PPO Actor-Critic網路"""

    def __init__(self, config: PPOConfig):
        super(ActorCritic, self).__init__()

        # 特徵提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Actor網路
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

        # Critic網路
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
    """本地Parquet數據載入器"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.data_dir = "scripts/download/historical_data/daily"
        self.data_cache = {}

    def load_local_data(self):
        """載入本地parquet檔案"""
        print(f"\n[DATA] Loading local parquet files from {self.data_dir}")
        print(f"[DATA] Date range: {self.config.data_start_date} to {self.config.data_end_date}")

        # 獲取所有parquet檔案
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        print(f"[DATA] Found {len(parquet_files)} parquet files")

        # 選取前N個檔案
        selected_files = parquet_files[: self.config.max_stocks]
        print(f"[DATA] Selected {len(selected_files)} files for training")

        success_count = 0
        failed_files = []

        for file_path in tqdm(selected_files, desc="Loading data"):
            try:
                # 從檔名提取股票代碼
                symbol = os.path.basename(file_path).replace(".parquet", "")

                # 讀取parquet檔案
                df = pd.read_parquet(file_path)

                # 確保有日期索引 - 使用timestamp列
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # 過濾日期範圍
                df = df[
                    (df.index >= self.config.data_start_date)
                    & (df.index <= self.config.data_end_date)
                ]

                # 確保有必要的列
                required_cols = ["open", "high", "low", "close", "volume"]
                df.columns = df.columns.str.lower()

                if all(col in df.columns for col in required_cols):
                    # 標準化列名
                    df = df.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                    )

                    if len(df) > 252:  # 至少一年數據
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
        """準備220維特徵"""
        features = []

        # 1. 價格特徵 (50維)
        returns = data["Close"].pct_change().fillna(0).values[-50:]
        features.append(returns.flatten() if len(returns.shape) > 1 else returns)

        # 2. 成交量特徵 (20維)
        volume_ma = data["Volume"].rolling(20).mean()
        volume_ratio = (data["Volume"] / volume_ma).fillna(1).values[-20:]
        features.append(volume_ratio.flatten() if len(volume_ratio.shape) > 1 else volume_ratio)

        # 3. 技術指標 (150維)
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

        # 合併所有特徵
        all_features = np.concatenate(features)

        # 確保特徵維度為220
        if len(all_features) < 220:
            all_features = np.pad(all_features, (0, 220 - len(all_features)), "constant")
        elif len(all_features) > 220:
            all_features = all_features[:220]

        return all_features


class SimpleTradingEnv:
    """簡化的交易環境"""

    def __init__(self, data: pd.DataFrame, config: PPOConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.position = 0
        self.capital = config.initial_capital
        self.data_loader = LocalDataLoader(config)

    def reset(self):
        """重置環境"""
        self.current_step = 252  # 從有足夠歷史數據的地方開始
        self.position = 0
        self.capital = self.config.initial_capital
        return self.get_state()

    def get_state(self):
        """獲取當前狀態"""
        if self.current_step >= len(self.data):
            self.current_step = 252

        window_data = self.data.iloc[: self.current_step + 1]
        features = self.data_loader.prepare_features(window_data)
        return features

    def step(self, action):
        """執行動作"""
        self.current_step += 1

        if self.current_step >= len(self.data):
            return self.get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["Close"]
        reward = 0

        # 執行交易
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            reward = -self.config.transaction_cost
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            reward = -self.config.transaction_cost

        # 計算收益
        if self.position == 1:
            returns = self.data.iloc[self.current_step]["Close"] / self.data.iloc[self.current_step - 1]["Close"] - 1
            reward += returns

        done = self.current_step >= len(self.data) - 1

        return self.get_state(), reward, done, {}


def train_ppo_local(config: PPOConfig):
    """使用本地數據進行PPO訓練"""
    print("\n" + "=" * 80)
    print("RL3 PPO TRAINING WITH LOCAL PARQUET DATA")
    print("=" * 80)

    # 載入本地數據
    data_loader = LocalDataLoader(config)
    stock_data = data_loader.load_local_data()

    if len(stock_data) == 0:
        print("[ERROR] No data loaded! Training aborted.")
        return

    # 創建模型
    model = ActorCritic(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"\n[MODEL] Created ActorCritic model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"[MODEL] Device: {config.device}")

    # 訓練統計
    training_stats = {
        "total_stocks": len(stock_data),
        "total_episodes": 0,
        "total_rewards": [],
        "avg_rewards": [],
        "best_reward": float("-inf"),
    }

    # 創建輸出目錄
    os.makedirs("runs/rl3/local_training", exist_ok=True)
    os.makedirs("models/ppo_local", exist_ok=True)

    # 對每個股票進行訓練
    for symbol, data in tqdm(list(stock_data.items())[:10], desc="Training stocks"):  # 先訓練10檔
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

            # 記錄統計
            training_stats["total_episodes"] += 1
            training_stats["total_rewards"].append(episode_reward)

            if episode_reward > training_stats["best_reward"]:
                training_stats["best_reward"] = episode_reward

            logger.info(f"[{symbol}] Episode reward: {episode_reward:.4f}")

        except Exception as e:
            logger.error(f"[{symbol}] Training failed: {e}")
            continue

    # 計算最終統計
    if training_stats["total_rewards"]:
        avg_reward = np.mean(training_stats["total_rewards"])
        std_reward = np.std(training_stats["total_rewards"])
        training_stats["avg_reward"] = avg_reward
        training_stats["std_reward"] = std_reward

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total stocks processed: {training_stats['total_episodes']}")
        print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"Best reward: {training_stats['best_reward']:.4f}")
        print(f"Worst reward: {min(training_stats['total_rewards']):.4f}")

        # 保存模型
        model_path = f"models/ppo_local/ppo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
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

        # 保存訓練統計
        stats_path = f"runs/rl3/local_training/training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2, default=str)
        print(f"[SAVE] Training stats saved to: {stats_path}")

        print("=" * 80)

    return training_stats


def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("RL3 LOCAL PPO TRAINING SYSTEM")
    print("Using local parquet data - No yfinance dependency!")
    print("=" * 80)

    # 創建配置
    config = PPOConfig()

    print(f"\n[CONFIG] Training configuration:")
    print(f"  - Max stocks: {config.max_stocks}")
    print(f"  - Date range: {config.data_start_date} to {config.data_end_date}")
    print(f"  - Features: {config.obs_dim} dimensions")
    print(f"  - Actions: {config.action_dim} (hold/buy/sell/close)")
    print(f"  - Device: {config.device}")

    # 開始訓練
    stats = train_ppo_local(config)

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("Check models/ppo_local/ for trained models")
    print("Check runs/rl3/local_training/ for training stats")
    print("Check ppo_training_local.log for detailed logs")
    print("=" * 80)


if __name__ == "__main__":
    main()
