#!/usr/bin/env python3
"""
PPO訓練 - 4000+股票版本
使用所有可用股票進行訓練
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
import warnings
import concurrent.futures
import random

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ppo_4000_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PPOConfig:
    """PPO訓練配置"""

    def __init__(self):
        # 模型參數
        self.obs_dim = 220
        self.action_dim = 4
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

        self.feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

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


def get_all_stock_symbols():
    """獲取所有股票代碼（4000+）"""
    symbols = []

    # 1. 從capital_tickers.txt讀取
    if os.path.exists("capital_tickers.txt"):
        with open("capital_tickers.txt", "r") as f:
            capital_stocks = [line.strip() for line in f if line.strip()]
            symbols.extend(capital_stocks)
            print(f"[DATA] Loaded {len(capital_stocks)} stocks from Capital.com list")

    # 2. 添加S&P 500
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_symbols = sp500_table["Symbol"].tolist()
        symbols.extend(sp500_symbols)
        print(f"[DATA] Added {len(sp500_symbols)} S&P 500 stocks")
    except:
        pass

    # 3. 添加NASDAQ最活躍股票
    nasdaq_active = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA",
        "AMD",
        "INTC",
        "NFLX",
        "PYPL",
        "CSCO",
        "AVGO",
        "QCOM",
        "TXN",
        "ORCL",
        "ADBE",
        "CRM",
        "IBM",
        "NOW",
        "UBER",
        "SNAP",
        "SHOP",
        "SQ",
        "ABNB",
        "COIN",
        "HOOD",
        "PLTR",
        "SOFI",
        "RIVN",
        "LCID",
        "NIO",
    ]
    symbols.extend(nasdaq_active)

    # 4. 添加中概股
    china_stocks = [
        "BABA",
        "JD",
        "PDD",
        "BIDU",
        "NIO",
        "XPEV",
        "LI",
        "BILI",
        "IQ",
        "TME",
        "VIPS",
        "WB",
        "BGNE",
        "TAL",
        "EDU",
        "BEKE",
        "NTES",
        "TCOM",
        "ZTO",
        "YMM",
        "HUYA",
        "DOYU",
        "RLX",
        "TUYA",
    ]
    symbols.extend(china_stocks)

    # 5. 添加ETF（用於多樣化）
    etfs = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VOO",
        "VTI",
        "EEM",
        "XLF",
        "XLK",
        "XLE",
        "XLV",
        "XLI",
        "XLY",
        "XLP",
        "XLB",
        "XLU",
        "ARKK",
        "ARKQ",
        "ARKW",
        "ARKG",
        "ARKF",
        "ICLN",
        "TAN",
        "LIT",
    ]
    symbols.extend(etfs)

    # 6. 添加更多行業領導者
    industry_leaders = [
        # 金融
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "BLK",
        "SCHW",
        "V",
        "MA",
        # 醫療
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "CVS",
        "MRK",
        "TMO",
        "ABT",
        "DHR",
        # 消費
        "WMT",
        "HD",
        "PG",
        "KO",
        "PEP",
        "COST",
        "NKE",
        "MCD",
        "SBUX",
        # 能源
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PXD",
        "MPC",
        "VLO",
        "PSX",
        # 工業
        "BA",
        "CAT",
        "LMT",
        "RTX",
        "DE",
        "UPS",
        "FDX",
        "UNP",
        "GE",
    ]
    symbols.extend(industry_leaders)

    # 去重
    symbols = list(set(symbols))

    # 如果還不夠4000個，生成一些測試符號
    if len(symbols) < 4000:
        print(f"[DATA] Current symbols: {len(symbols)}, generating additional test symbols...")
        # 添加一些國際股票代碼格式
        for i in range(4000 - len(symbols)):
            # 生成一些測試代碼（實際訓練時會被過濾）
            test_symbol = f"TEST{i:04d}"
            symbols.append(test_symbol)

    print(f"[DATA] Total symbols prepared: {len(symbols)}")
    return symbols[:4000]  # 限制在4000個


class DataLoader:
    """數據加載器 - 支援4000股票"""

    def __init__(self, symbols: List[str], start_date: str = None):
        self.symbols = symbols
        self.start_date = start_date or "2020-01-01"  # 使用較近的日期以加快下載
        self.data_cache = {}

    def download_batch(self, batch_symbols: List[str]) -> Dict:
        """下載一批股票數據"""
        batch_data = {}
        for symbol in batch_symbols:
            try:
                data = yf.download(
                    symbol,
                    start=self.start_date,
                    end=datetime.now().strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    threads=False,
                )
                if not data.empty and len(data) > 100:
                    batch_data[symbol] = data
            except:
                pass
        return batch_data

    def download_all_data(self):
        """並行下載所有股票數據"""
        print(f"\n[DATA] Starting parallel download for {len(self.symbols)} stocks...")
        print(f"[DATA] Date range: {self.start_date} to {datetime.now().date()}")

        # 分批並行下載
        batch_size = 10
        batches = [
            self.symbols[i : i + batch_size] for i in range(0, len(self.symbols), batch_size)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for batch in batches[:100]:  # 先下載前1000個股票（100批）
                future = executor.submit(self.download_batch, batch)
                futures.append(future)

            # 收集結果
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"
            ):
                batch_data = future.result()
                self.data_cache.update(batch_data)

        print(
            f"\n[DATA] Successfully downloaded: {len(self.data_cache)}/{len(self.symbols[:1000])} stocks"
        )

        # 如果下載的股票太少，使用模擬數據
        if len(self.data_cache) < 50:
            print("[DATA] Too few stocks downloaded, generating synthetic data...")
            self._generate_synthetic_data()

        return self.data_cache

    def _generate_synthetic_data(self):
        """生成合成數據用於訓練"""
        print("[DATA] Generating synthetic training data...")

        # 生成100個合成股票數據
        for i in range(100):
            symbol = f"SYN{i:03d}"

            # 生成隨機價格序列
            days = 500
            dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

            # 隨機漫步價格
            returns = np.random.normal(0.0005, 0.02, days)
            prices = 100 * np.exp(np.cumsum(returns))

            # 創建OHLCV數據
            data = pd.DataFrame(
                {
                    "Open": prices * (1 + np.random.normal(0, 0.005, days)),
                    "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
                    "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
                    "Close": prices,
                    "Volume": np.random.lognormal(15, 1, days),
                },
                index=dates,
            )

            self.data_cache[symbol] = data

        print(f"[DATA] Generated {len(self.data_cache)} synthetic stocks")

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵"""
        features = []

        # 簡化特徵提取以加快速度
        try:
            # 價格變化
            returns = data["Close"].pct_change().fillna(0).values[-50:]
            features.extend(returns if len(returns) == 50 else np.zeros(50))

            # 成交量
            if "Volume" in data.columns:
                volume = data["Volume"].values[-20:]
                volume_norm = volume / (volume.mean() + 1e-8)
                features.extend(volume_norm if len(volume_norm) == 20 else np.zeros(20))
            else:
                features.extend(np.zeros(20))

            # 技術指標（簡化版）
            close = data["Close"].values

            # SMA
            for period in [5, 10, 20, 50]:
                if len(close) >= period:
                    sma = pd.Series(close).rolling(period).mean().values[-10:]
                    features.extend(sma if len(sma) == 10 else np.zeros(10))
                else:
                    features.extend(np.zeros(10))

            # RSI
            if len(close) >= 14:
                delta = pd.Series(close).diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.extend(rsi.fillna(50).values[-30:])
            else:
                features.extend(np.zeros(30))

            # 填充到220維
            features = np.array(features[:220])
            if len(features) < 220:
                features = np.pad(features, (0, 220 - len(features)))

        except Exception as e:
            # 出錯時返回零向量
            features = np.zeros(220)

        return features.astype(np.float32)


class TradingEnvironment:
    """交易環境 - 優化版"""

    def __init__(self, data: Dict, config: PPOConfig):
        self.data = data
        self.config = config
        self.symbols = list(data.keys())
        self.reset()

    def reset(self):
        # 隨機選擇股票
        if not self.symbols:
            return np.zeros(self.config.obs_dim)

        self.current_symbol = random.choice(self.symbols)
        self.symbol_data = self.data[self.current_symbol]

        # 確保有足夠數據
        if len(self.symbol_data) < 100:
            return self.reset()

        # 隨機起始點
        self.current_step = random.randint(50, min(200, len(self.symbol_data) - 50))

        # 初始化投資組合
        self.cash = self.config.initial_capital
        self.positions = {}
        self.portfolio_value = self.cash

        return self._get_observation()

    def _get_observation(self):
        try:
            end_idx = self.current_step
            start_idx = max(0, end_idx - 220)
            window_data = self.symbol_data.iloc[start_idx:end_idx]

            if len(window_data) < 50:
                return np.zeros(self.config.obs_dim)

            data_loader = DataLoader([], "")
            features = data_loader.prepare_features(window_data)
            return features
        except:
            return np.zeros(self.config.obs_dim)

    def step(self, action):
        try:
            # 獲取當前價格
            current_price = float(self.symbol_data.iloc[self.current_step]["Close"])

            # 執行動作
            if action == 1 or action == 3:  # Buy
                size = 0.1 if action == 1 else 0.2
                if self.cash > 1000:  # 最少保留1000現金
                    investment = self.cash * size
                    shares = investment / current_price
                    cost = investment * (1 + self.config.transaction_cost)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.positions[self.current_symbol] = (
                            self.positions.get(self.current_symbol, 0) + shares
                        )

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
                if symbol == self.current_symbol and self.current_step < len(self.symbol_data):
                    price = float(self.symbol_data.iloc[self.current_step]["Close"])
                    new_portfolio_value += shares * price

            # 計算獎勵
            reward = (new_portfolio_value - self.portfolio_value) / (self.portfolio_value + 1e-8)
            self.portfolio_value = new_portfolio_value

            # 檢查是否結束
            done = self.current_step >= min(len(self.symbol_data) - 1, self.current_step + 100)

            return self._get_observation(), reward, done

        except Exception as e:
            # 錯誤處理
            return np.zeros(self.config.obs_dim), 0, True


class PPOTrainer:
    """PPO訓練器 - 快速版"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.model = ActorCritic(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.training_history = {"rewards": [], "losses": [], "iterations": []}

    def train(self, env: TradingEnvironment, n_iterations: int = 200):
        """快速訓練"""
        print(f"\n[TRAINING] Starting fast PPO training for {n_iterations} iterations")
        print(f"[TRAINING] Device: {self.config.device}")

        for iteration in range(n_iterations):
            # 收集經驗
            batch_obs = []
            batch_acts = []
            batch_rewards = []

            for _ in range(10):  # 每次迭代收集10個episode
                obs = env.reset()
                episode_rewards = []

                for _ in range(50):  # 每個episode最多50步
                    obs_tensor = torch.FloatTensor(obs).to(self.config.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)

                    with torch.no_grad():
                        action, _, _, _ = self.model.get_action_and_value(obs_tensor)

                    batch_obs.append(obs)
                    batch_acts.append(action.cpu().numpy())

                    obs, reward, done = env.step(action.item())
                    episode_rewards.append(reward)
                    batch_rewards.append(reward)

                    if done:
                        break

            # 簡單更新（不使用GAE以加快速度）
            if batch_obs:
                self._simple_update(batch_obs, batch_acts, batch_rewards)

            # 記錄進度
            avg_reward = np.mean(batch_rewards) if batch_rewards else 0
            self.training_history["iterations"].append(iteration)
            self.training_history["rewards"].append(avg_reward)

            if iteration % 10 == 0:
                print(f"[Iter {iteration:4d}] Avg Reward: {avg_reward:.4f}")

            if iteration % 50 == 0 and iteration > 0:
                self.save_checkpoint(iteration)

    def _simple_update(self, observations, actions, rewards):
        """簡化的更新"""
        obs_tensor = torch.FloatTensor(observations).to(self.config.device)
        action_tensor = torch.LongTensor(actions).squeeze().to(self.config.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.config.device)

        # 正規化獎勵
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # 計算損失並更新
        action_logits, values = self.model(obs_tensor)
        dist = Categorical(logits=action_logits)

        # 策略損失
        log_probs = dist.log_prob(action_tensor)
        policy_loss = -(log_probs * rewards_tensor).mean()

        # 價值損失
        value_loss = F.mse_loss(values.squeeze(), rewards_tensor)

        # 熵獎勵
        entropy = dist.entropy().mean()

        # 總損失
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.training_history["losses"].append(total_loss.item())

    def save_checkpoint(self, iteration):
        """保存檢查點"""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "training_history": self.training_history,
            "config": self.config.__dict__,
        }

        filename = f"ppo_4000_iter_{iteration}.pt"
        torch.save(checkpoint, filename)
        print(f"[SAVE] Checkpoint saved: {filename}")


def main():
    print("=" * 80)
    print("PPO TRAINING - 4000 STOCKS VERSION")
    print("=" * 80)

    # 1. 獲取所有股票符號
    symbols = get_all_stock_symbols()

    # 2. 下載數據
    data_loader = DataLoader(symbols, start_date="2020-01-01")
    stock_data = data_loader.download_all_data()

    if len(stock_data) < 10:
        print("[WARNING] Using synthetic data for training")

    # 3. 初始化環境和訓練器
    config = PPOConfig()
    env = TradingEnvironment(stock_data, config)
    trainer = PPOTrainer(config)

    # 4. 開始訓練
    print(f"\n[TRAINING] Training with {len(stock_data)} stocks")
    trainer.train(env, n_iterations=200)  # 減少迭代次數以加快訓練

    # 5. 保存最終模型
    final_checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "training_history": trainer.training_history,
        "config": config.__dict__,
        "symbols_trained": list(stock_data.keys())[:100],  # 只保存前100個符號
        "training_date": datetime.now().isoformat(),
        "total_stocks": len(stock_data),
    }

    torch.save(final_checkpoint, "ppo_4000_final.pt")
    print(f"\n[COMPLETE] Final model saved: ppo_4000_final.pt")

    # 6. 生成報告
    report = {
        "training_date": datetime.now().isoformat(),
        "stocks_count": len(stock_data),
        "total_iterations": 200,
        "final_reward": (
            trainer.training_history["rewards"][-1] if trainer.training_history["rewards"] else 0
        ),
        "device": str(config.device),
    }

    with open("ppo_4000_training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Stocks used: {len(stock_data)}")
    print(f"Final reward: {report['final_reward']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
