#!/usr/bin/env python3
"""
完整PPO訓練腳本 - 使用所有歷史數據訓練4000檔股票
Full PPO Training with Maximum Historical Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ppo_training_full.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PPOConfig:
    """PPO訓練配置"""

    def __init__(self):
        # 模型參數
        self.obs_dim = 220  # 特徵維度
        self.action_dim = 4  # 動作空間
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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(self, obs):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, action_log_prob, value, entropy


class DataLoader:
    """數據加載器 - 支援4000檔股票"""

    def __init__(self, symbols: List[str], start_date: str = None):
        self.symbols = symbols
        self.start_date = start_date or "2010-01-01"  # 使用最長歷史
        self.data_cache = {}

    def download_all_data(self):
        """下載所有股票的歷史數據"""
        print(f"\n[DATA] Downloading historical data for {len(self.symbols)} stocks...")
        print(f"[DATA] Date range: {self.start_date} to {datetime.now().date()}")

        failed_symbols = []
        success_count = 0

        # 批量下載以提高效率
        batch_size = 100
        for i in tqdm(range(0, len(self.symbols), batch_size), desc="Downloading batches"):
            batch_symbols = self.symbols[i : i + batch_size]
            batch_str = " ".join(batch_symbols)

            try:
                # 下載整批數據
                yf.download(
                    batch_str,
                    start=self.start_date,
                    end=datetime.now().strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )

                # 儲存到緩存
                if len(batch_symbols) == 1:
                    # 單一股票
                    if not data.empty:
                        self.data_cache[batch_symbols[0]] = data
                        success_count += 1
                    else:
                        failed_symbols.append(batch_symbols[0])
                else:
                    # 多股票
                    for symbol in batch_symbols:
                        try:
                            if symbol in data.columns.levels[1]:
                                symbol_data = data.xs(symbol, level=1, axis=1)
                                if not symbol_data.empty:
                                    self.data_cache[symbol] = symbol_data
                                    success_count += 1
                                else:
                                    failed_symbols.append(symbol)
                            else:
                                failed_symbols.append(symbol)
                        except Exception:
                            failed_symbols.append(symbol)

            except Exception as e:
                logger.error(f"Batch download error: {e}")
                failed_symbols.extend(batch_symbols)

        print(f"\n[DATA] Successfully downloaded: {success_count}/{len(self.symbols)} stocks")
        if failed_symbols:
            print(f"[DATA] Failed symbols: {failed_symbols[:10]}...")

        return self.data_cache

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵"""
        features = []

        # 1. 價格特徵
        features.append(data["Close"].pct_change().fillna(0).values[-50:])  # 50日收益率

        # 2. 成交量特徵
        volume_ma = data["Volume"].rolling(20).mean()
        volume_ratio = (data["Volume"] / volume_ma).fillna(1).values[-20:]
        features.append(volume_ratio)

        # 3. 技術指標
        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50).values[-30:])

        # MACD
        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features.append(macd.fillna(0).values[-30:])
        features.append(signal.fillna(0).values[-30:])

        # 布林通道
        bb_ma = data["Close"].rolling(20).mean()
        bb_std = data["Close"].rolling(20).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_position = ((data["Close"] - bb_lower) / (bb_upper - bb_lower)).fillna(0.5)
        features.append(bb_position.values[-30:])

        # 移動平均線
        for period in [5, 10, 20, 50]:
            ma = data["Close"].rolling(period).mean()
            ma_ratio = (data["Close"] / ma).fillna(1)
            features.append(ma_ratio.values[-10:])

        # 展平所有特徵
        all_features = np.concatenate(features)

        # 確保維度為220
        if len(all_features) < 220:
            all_features = np.pad(all_features, (0, 220 - len(all_features)))
        elif len(all_features) > 220:
            all_features = all_features[:220]

        return all_features


class TradingEnvironment:
    """交易環境"""

    def __init__(self, data: Dict, config: PPOConfig):
        self.data = data
        self.config = config
        self.reset()

    def reset(self):
        # 隨機選擇股票和起始時間
        self.current_symbol = np.random.choice(list(self.data.keys()))
        self.symbol_data = self.data[self.current_symbol]

        # 確保有足夠的歷史數據
        min_history = 100
        if len(self.symbol_data) < min_history + 100:
            return self.reset()  # 重新選擇

        # 隨機起始點
        self.current_step = np.random.randint(min_history, len(self.symbol_data) - 100)

        # 初始化投資組合
        self.cash = self.config.initial_capital
        self.positions = {}
        self.portfolio_value = self.cash

        return self._get_observation()

    def _get_observation(self):
        # 獲取當前特徵
        end_idx = min(self.current_step, len(self.symbol_data))
        if end_idx < 220:
            # 數據不足，返回零向量
            return np.zeros(self.config.obs_dim)

        window_data = self.symbol_data.iloc[max(0, end_idx - 220) : end_idx]

        # 準備特徵
        data_loader = DataLoader([], "")
        features = data_loader.prepare_features(window_data)

        return features

    def step(self, action):
        # 獲取當前價格
        current_price = self.symbol_data.iloc[self.current_step]["Close"]

        # 執行動作
        # 0: Hold, 1: Buy, 2: Sell, 3: Strong Buy/Sell
        reward = 0

        if action == 1 or action == 3:  # Buy
            size = 0.1 if action == 1 else 0.2  # 加倍倉位for strong signal
            if self.cash > 0:
                shares = (self.cash * size) / current_price
                cost = shares * current_price * (1 + self.config.transaction_cost)
                if cost <= self.cash:
                    self.cash -= cost
                    if self.current_symbol in self.positions:
                        self.positions[self.current_symbol] += shares
                    else:
                        self.positions[self.current_symbol] = shares

        elif action == 2:  # Sell
            if self.current_symbol in self.positions:
                shares = self.positions[self.current_symbol]
                revenue = shares * current_price * (1 - self.config.transaction_cost)
                self.cash += revenue
                del self.positions[self.current_symbol]

        # 前進一步
        self.current_step += 1

        # 計算新的投資組合價值
        new_portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol == self.current_symbol:
                new_portfolio_value += shares * self.symbol_data.iloc[self.current_step]["Close"]

        # 計算獎勵
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.portfolio_value = new_portfolio_value

        # 檢查是否結束
        done = self.current_step >= len(self.symbol_data) - 1

        return self._get_observation(), reward, done


class PPOTrainer:
    """PPO訓練器"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.model = ActorCritic(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.training_history = {"rewards": [], "losses": [], "iterations": []}

    def train(self, env: TradingEnvironment, n_iterations: int = 1000):
        """訓練PPO模型"""
        print(f"\n[TRAINING] Starting PPO training for {n_iterations} iterations")
        print(f"[TRAINING] Device: {self.config.device}")

        for iteration in range(n_iterations):
            # 收集軌跡
            observations = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []

            env.reset()

            for _ in range(self.config.n_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)

                with torch.no_grad():
                    action, log_prob, value, _ = self.model.get_action_and_value(obs_tensor)

                observations.append(obs)
                actions.append(action.cpu().numpy())
                log_probs.append(log_prob.cpu().numpy())
                values.append(value.cpu().numpy())

                obs, reward, done = env.step(action.item())
                rewards.append(reward)
                dones.append(done)

                if done:
                    env.reset()

            # 計算優勢和回報
            advantages, returns = self._compute_gae(rewards, values, dones)

            # PPO更新
            total_loss = self._ppo_update(observations, actions, log_probs, advantages, returns)

            # 記錄訓練歷史
            self.training_history["iterations"].append(iteration)
            self.training_history["rewards"].append(np.mean(rewards))
            self.training_history["losses"].append(total_loss)

            # 打印進度
            if iteration % 10 == 0:
                print(
                    f"[Iter {iteration:4d}] Reward: {np.mean(rewards):.4f}, Loss: {total_loss:.4f}"
                )

            # 保存檢查點
            if iteration % 50 == 0 and iteration > 0:
                self.save_checkpoint(iteration)

    def _compute_gae(self, rewards, values, dones):
        """計算廣義優勢估計"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        returns = advantages + np.array(values).squeeze()

        return advantages, returns

    def _ppo_update(self, observations, actions, old_log_probs, advantages, returns):
        """PPO策略更新"""
        # 轉換為張量
        obs_tensor = torch.FloatTensor(observations).to(self.config.device)
        action_tensor = torch.LongTensor(actions).squeeze().to(self.config.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).squeeze().to(self.config.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.config.device)
        returns_tensor = torch.FloatTensor(returns).to(self.config.device)

        # 正規化優勢
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # 多個epoch更新
        for _ in range(self.config.n_epochs):
            # 前向傳播
            action_logits, values = self.model(obs_tensor)
            dist = Categorical(logits=action_logits)

            # 計算損失
            new_log_probs = dist.log_prob(action_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # 策略損失
            surr1 = ratio * advantages_tensor
            surr2 = (
                torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * advantages_tensor
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # 價值損失
            value_loss = F.mse_loss(values.squeeze(), returns_tensor)

            # 熵損失
            entropy_loss = -dist.entropy().mean()

            # 總損失
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

            # 反向傳播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return total_loss.item()

    def save_checkpoint(self, iteration):
        """保存模型檢查點"""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "config": self.config.__dict__,
        }

        filename = f"ppo_full_iter_{iteration}.pt"
        torch.save(checkpoint, filename)
        print(f"[SAVE] Checkpoint saved: {filename}")


def main():
    """主訓練流程"""
    print("=" * 80)
    print("PPO FULL TRAINING - MAXIMUM HISTORICAL DATA")
    print("=" * 80)

    # 1. 載入股票列表
    print("\n[INIT] Loading stock symbols...")
    []

    # 從文件載入4000檔股票
    if os.path.exists("data/all_symbols.txt"):
        with open("data/all_symbols.txt", "r") as f:
            [line.strip() for line in f.readlines() if line.strip()]
            symbols[:4000]  # 限制4000檔
    else:
        # 使用預設股票列表
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]

    print(f"[INIT] Loaded {len(symbols)} stock symbols")

    # 2. 下載歷史數據
    data_loader = DataLoader(symbols, start_date="2010-01-01")  # 使用最長歷史
    stock_data = data_loader.download_all_data()

    if len(stock_data) < 10:
        print("[ERROR] Insufficient data downloaded")
        return

    # 3. 初始化訓練環境
    config = PPOConfig()
    env = TradingEnvironment(stock_data, config)

    # 4. 初始化訓練器
    trainer = PPOTrainer(config)

    # 5. 開始訓練
    print(f"\n[TRAINING] Training with {len(stock_data)} stocks")
    print(f"[TRAINING] Total data points: ~{sum(len(df) for df in stock_data.values())}")

    # 訓練1000次迭代（可調整）
    trainer.train(env, n_iterations=1000)

    # 6. 保存最終模型
    final_checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "training_history": trainer.training_history,
        "config": config.__dict__,
        "symbols_trained": list(stock_data.keys()),
        "training_date": datetime.now().isoformat(),
        "data_range": f"2010-01-01 to {datetime.now().date()}",
    }

    torch.save(final_checkpoint, "ppo_full_final.pt")
    print("\n[COMPLETE] Final model saved: ppo_full_final.pt")

    # 7. 生成訓練報告
    {
        "training_date": datetime.now().isoformat(),
        "stocks_count": len(stock_data),
        "data_range": f"2010-01-01 to {datetime.now().date()}",
        "total_iterations": 1000,
        "final_reward": (
            trainer.training_history["rewards"][-1] if trainer.training_history["rewards"] else 0
        ),
        "device": str(config.device),
    }

    with open("ppo_training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Stocks trained: {len(stock_data)}")
    print("Data range: 2010-2025 (15 years)")
    print(f"Final reward: {report['final_reward']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
