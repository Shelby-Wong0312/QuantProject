#!/usr/bin/env python3
"""
RL3 PPO Training with Mixed Market Regimes (Bull + Bear Markets)
混合市場環境訓練：包含牛市和熊市數據，降低過度交易
Date: 2025-11-19

改進重點：
1. 擴展訓練期間：2015-2025（包含 2021-2022 熊市）
2. 提高交易成本：0.2%（降低交易頻率）
3. 添加換手率懲罰：lambda_turnover=0.001
4. 增加訓練股票數：100->200 檔
5. 熊市期間加權訓練（2021-2022 重複 2x）
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
        logging.FileHandler("ppo_training_mixed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PPOMixedConfig:
    """PPO混合訓練配置 - 針對熊牛市通吃優化"""

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

        # 環境參數 - 優化以降低過度交易
        self.initial_capital = 100000
        self.max_positions = 20
        self.transaction_cost = 0.002  # ⬆️ 從 0.1% 提高到 0.2%
        self.turnover_penalty = 0.001  # ✨ 新增：換手率懲罰

        # 數據參數 - 擴展至包含熊市
        self.data_start_date = "2015-01-01"  # ⬆️ 包含更多歷史
        self.data_end_date = "2025-08-08"    # ⬆️ 延伸至最新數據
        self.max_stocks = 200  # ⬆️ 從 100 增加到 200 檔股票

        # 熊市期間（用於加權訓練）
        self.bear_market_start = "2021-01-01"
        self.bear_market_end = "2022-12-31"
        self.bear_market_weight = 2.0  # 熊市數據重複 2 次

        # 設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("\n" + "=" * 80)
        print("PPO MIXED MARKET CONFIG")
        print("=" * 80)
        print(f"Training Period: {self.data_start_date} to {self.data_end_date}")
        print(f"Bear Market Focus: {self.bear_market_start} to {self.bear_market_end} (weight: {self.bear_market_weight}x)")
        print(f"Transaction Cost: {self.transaction_cost*100:.2f}% (vs 0.1% baseline)")
        print(f"Turnover Penalty: {self.turnover_penalty} (new)")
        print(f"Max Stocks: {self.max_stocks}")
        print(f"Device: {self.device}")
        print("=" * 80 + "\n")


class ActorCritic(nn.Module):
    """PPO Actor-Critic網路（與原版相同）"""

    def __init__(self, config: PPOMixedConfig):
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
    """本地Parquet數據載入器 - 支持熊市加權"""

    def __init__(self, config: PPOMixedConfig):
        self.config = config
        self.data_dir = "scripts/download/historical_data/daily"
        self.data_cache = {}
        self.bear_market_symbols = []  # 記錄熊市期間可用的股票

    def load_local_data(self):
        """載入本地parquet檔案 - 優先選擇包含熊市數據的股票"""
        print(f"\n[DATA] Loading local parquet files from {self.data_dir}")
        print(f"[DATA] Date range: {self.config.data_start_date} to {self.config.data_end_date}")
        print(f"[DATA] Bear market focus: {self.config.bear_market_start} to {self.config.bear_market_end}")

        # 獲取所有parquet檔案
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        print(f"[DATA] Found {len(parquet_files)} parquet files")

        success_count = 0
        failed_files = []
        bear_market_count = 0

        for file_path in tqdm(parquet_files[: self.config.max_stocks * 2], desc="Loading data"):
            try:
                # 從檔名提取股票代碼
                symbol = os.path.basename(file_path).replace(".parquet", "").replace("_daily", "")

                # 讀取parquet檔案
                df = pd.read_parquet(file_path)

                # 確保有日期索引
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

                        # 檢查是否包含熊市數據
                        bear_data = df[
                            (df.index >= self.config.bear_market_start)
                            & (df.index <= self.config.bear_market_end)
                        ]
                        if len(bear_data) > 200:  # 熊市期間至少 200 天數據
                            self.bear_market_symbols.append(symbol)
                            bear_market_count += 1

                        # 達到目標數量就停止
                        if success_count >= self.config.max_stocks:
                            break
                    else:
                        failed_files.append(f"{symbol} (insufficient data: {len(df)} days)")
                else:
                    failed_files.append(f"{symbol} (missing columns)")

            except Exception as e:
                failed_files.append(f"{os.path.basename(file_path)} ({str(e)[:50]})")

        print(
            f"\n[DATA] Successfully loaded: {success_count}/{min(len(parquet_files), self.config.max_stocks*2)} files"
        )
        print(f"[DATA] Stocks with bear market data: {bear_market_count}")
        if failed_files[:3]:
            print(f"[DATA] Sample failed files: {failed_files[:3]}")

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


class EnhancedTradingEnv:
    """增強的交易環境 - 包含換手率懲罰"""

    def __init__(self, data: pd.DataFrame, config: PPOMixedConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.position = 0
        self.last_position = 0
        self.capital = config.initial_capital
        self.data_loader = LocalDataLoader(config)
        self.trade_count = 0

    def reset(self):
        """重置環境"""
        self.current_step = 252  # 從有足夠歷史數據的地方開始
        self.position = 0
        self.last_position = 0
        self.capital = self.config.initial_capital
        self.trade_count = 0
        return self.get_state()

    def get_state(self):
        """獲取當前狀態"""
        if self.current_step >= len(self.data):
            self.current_step = 252

        window_data = self.data.iloc[: self.current_step + 1]
        features = self.data_loader.prepare_features(window_data)
        return features

    def step(self, action):
        """執行動作 - 包含換手率懲罰"""
        self.current_step += 1

        if self.current_step >= len(self.data):
            return self.get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["Close"]
        reward = 0
        self.last_position = self.position

        # 執行交易
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.trade_count += 1
            # 交易成本 + 換手率懲罰
            reward = -(self.config.transaction_cost + self.config.turnover_penalty)
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.trade_count += 1
            # 交易成本 + 換手率懲罰
            reward = -(self.config.transaction_cost + self.config.turnover_penalty)

        # 計算收益
        if self.position == 1:
            returns = (
                self.data.iloc[self.current_step]["Close"]
                / self.data.iloc[self.current_step - 1]["Close"]
                - 1
            )
            reward += returns

        done = self.current_step >= len(self.data) - 1

        return self.get_state(), reward, done, {"trade_count": self.trade_count}


def train_ppo_mixed(config: PPOMixedConfig):
    """使用混合市場數據進行PPO訓練"""
    print("\n" + "=" * 80)
    print("RL3 PPO TRAINING WITH MIXED MARKET REGIMES (BULL + BEAR)")
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

    print(f"\n[MODEL] Created ActorCritic model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[MODEL] Device: {config.device}")

    # 訓練統計
    training_stats = {
        "total_stocks": len(stock_data),
        "bear_market_stocks": len(data_loader.bear_market_symbols),
        "total_episodes": 0,
        "total_rewards": [],
        "avg_rewards": [],
        "trade_counts": [],
        "best_reward": float("-inf"),
        "config": {
            "transaction_cost": config.transaction_cost,
            "turnover_penalty": config.turnover_penalty,
            "bear_market_weight": config.bear_market_weight,
            "date_range": f"{config.data_start_date} to {config.data_end_date}",
        },
    }

    # 創建輸出目錄
    os.makedirs("runs/rl3/mixed_training", exist_ok=True)
    os.makedirs("models/ppo_local", exist_ok=True)

    # 準備訓練序列：熊市股票訓練2次，其他訓練1次
    training_sequence = []
    for symbol, data in stock_data.items():
        training_sequence.append((symbol, data))
        # 如果是熊市股票，再加一次
        if symbol in data_loader.bear_market_symbols:
            training_sequence.append((symbol, data))

    print(f"\n[TRAINING] Total training episodes: {len(training_sequence)}")
    print(f"[TRAINING] Bear market episodes repeated: {len(data_loader.bear_market_symbols)}")

    # 對序列進行訓練
    for symbol, data in tqdm(training_sequence, desc="Training episodes"):
        try:
            env = EnhancedTradingEnv(data, config)
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

                next_state, reward, done, info = env.step(action.item())

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
            training_stats["trade_counts"].append(info.get("trade_count", 0))

            if episode_reward > training_stats["best_reward"]:
                training_stats["best_reward"] = episode_reward

            is_bear = "🐻" if symbol in data_loader.bear_market_symbols else ""
            logger.info(
                f"[{symbol}{is_bear}] Episode reward: {episode_reward:.4f}, Trades: {info.get('trade_count', 0)}"
            )

        except Exception as e:
            logger.error(f"[{symbol}] Training failed: {e}")
            continue

    # 計算最終統計
    if training_stats["total_rewards"]:
        avg_reward = np.mean(training_stats["total_rewards"])
        std_reward = np.std(training_stats["total_rewards"])
        avg_trades = np.mean(training_stats["trade_counts"])
        training_stats["avg_reward"] = avg_reward
        training_stats["std_reward"] = std_reward
        training_stats["avg_trades"] = avg_trades

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total episodes processed: {training_stats['total_episodes']}")
        print(f"Unique stocks: {training_stats['total_stocks']}")
        print(f"Bear market stocks (2x training): {training_stats['bear_market_stocks']}")
        print(f"\nAverage reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"Average trades per episode: {avg_trades:.2f}")
        print(f"Best reward: {training_stats['best_reward']:.4f}")
        print(f"Worst reward: {min(training_stats['total_rewards']):.4f}")

        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/ppo_local/ppo_model_mixed_{timestamp}.pt"
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
        stats_path = f"runs/rl3/mixed_training/training_stats_{timestamp}.json"
        with open(stats_path, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_stats = {}
            for key, value in training_stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_stats[key] = value.tolist()
                elif isinstance(value, list):
                    serializable_stats[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
                else:
                    serializable_stats[key] = value

            json.dump(serializable_stats, f, indent=2)
        print(f"[SAVE] Training stats saved to: {stats_path}")

        print("=" * 80)

    return training_stats, model_path


def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("RL3 MIXED MARKET PPO TRAINING SYSTEM")
    print("Training on BULL + BEAR markets to reduce overtrading")
    print("=" * 80)

    # 創建配置
    config = PPOMixedConfig()

    # 開始訓練
    stats, model_path = train_ppo_mixed(config)

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Model saved: {model_path}")
    print("✓ Check runs/rl3/mixed_training/ for training stats")
    print("✓ Check ppo_training_mixed.log for detailed logs")
    print("\n📊 NEXT STEPS:")
    print("1. Run backtest on 2021-2022 with new model")
    print("2. Run backtest on 2023-2025 with new model")
    print("3. Compare with baseline model performance")
    print("=" * 80)


if __name__ == "__main__":
    main()
