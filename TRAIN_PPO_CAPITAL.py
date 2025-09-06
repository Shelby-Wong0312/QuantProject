#!/usr/bin/env python3
"""
PPO訓練 - 使用Capital.com真實4446個股票
PPO Training with Real Capital.com Stocks
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
import random
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ppo_capital_training.log'),
        logging.StreamHandler()
    ]
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    """PPO Actor-Critic網路"""
    def __init__(self, config: PPOConfig):
        super(ActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
            nn.Linear(config.hidden_dim, config.action_dim)
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
            nn.Linear(128, 1)
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

class CapitalStockDataLoader:
    """Capital.com股票數據加載器"""
    def __init__(self, use_all: bool = True):
        self.use_all = use_all
        self.data_cache = {}
        self.start_date = "2020-01-01"  # 使用3年數據
        
    def load_capital_stocks(self):
        """載入Capital.com股票列表"""
        if os.path.exists('capital_real_tickers.txt'):
            with open('capital_real_tickers.txt', 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            print(f"[DATA] Loaded {len(tickers)} Capital.com tickers")
            print(f"[DATA] Using ALL {len(tickers)} stocks for training (no sampling)")
            
            return tickers
        else:
            print("[ERROR] capital_real_tickers.txt not found")
            return []
    
    def download_stock_data(self, tickers: List[str]):
        """下載股票數據"""
        print(f"[DATA] Downloading data for {len(tickers)} stocks...")
        print(f"[DATA] Date range: {self.start_date} to {datetime.now().date()}")
        print(f"[DATA] This will take approximately {len(tickers) // 60} minutes...")
        
        success_count = 0
        failed_tickers = []
        
        # 批量下載以提高效率
        batch_size = 10
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        
        for batch in tqdm(batches, desc="Downloading batches"):
            for ticker in batch:
                try:
                    data = yf.download(
                        ticker,
                        start=self.start_date,
                        end=datetime.now().strftime('%Y-%m-%d'),
                        interval='1d',
                        progress=False,
                        threads=False
                    )
                    
                    if not data.empty and len(data) > 252:  # 至少一年數據
                        self.data_cache[ticker] = data
                        success_count += 1
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    failed_tickers.append(ticker)
            
            # 每10批（100個股票）暫停一下
            if len(self.data_cache) % 100 == 0 and len(self.data_cache) > 0:
                print(f"[PROGRESS] Downloaded {success_count} stocks so far...")
                import time
                time.sleep(0.5)
        
        print(f"[DATA] Successfully downloaded: {success_count}/{len(tickers)} stocks")
        
        # 如果下載太少，生成合成數據
        if success_count < 50:
            print("[DATA] Too few stocks, generating synthetic data...")
            self._generate_synthetic_data()
        
        return self.data_cache
    
    def _generate_synthetic_data(self):
        """生成合成訓練數據"""
        for i in range(100):
            symbol = f"SYNTH_{i:03d}"
            
            # 生成價格序列
            days = 750  # 3年數據
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # 隨機漫步
            returns = np.random.normal(0.0005, 0.02, days)
            prices = 100 * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, days)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
                'Close': prices,
                'Volume': np.random.lognormal(15, 1, days)
            }, index=dates)
            
            self.data_cache[symbol] = data
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵"""
        features = []
        
        try:
            # 1. 收益率 (50維)
            returns = data['Close'].pct_change().fillna(0).values[-50:]
            features.extend(returns if len(returns) == 50 else np.zeros(50))
            
            # 2. 成交量 (20維)
            if 'Volume' in data.columns:
                volume = data['Volume'].values[-20:]
                volume_norm = volume / (volume.mean() + 1e-8)
                features.extend(volume_norm if len(volume_norm) == 20 else np.zeros(20))
            else:
                features.extend(np.zeros(20))
            
            # 3. RSI (30維)
            close = data['Close'].values
            if len(close) >= 14:
                delta = pd.Series(close).diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.extend(rsi.fillna(50).values[-30:])
            else:
                features.extend(np.zeros(30))
            
            # 4. 移動平均 (40維)
            for period in [5, 10, 20, 50]:
                if len(close) >= period:
                    sma = pd.Series(close).rolling(period).mean().values[-10:]
                    features.extend(sma if len(sma) == 10 else np.zeros(10))
                else:
                    features.extend(np.zeros(10))
            
            # 5. MACD (60維)
            if len(close) >= 26:
                exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
                exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                features.extend(macd.values[-30:])
                features.extend(signal.values[-30:])
            else:
                features.extend(np.zeros(60))
            
            # 6. 布林帶 (20維)
            if len(close) >= 20:
                ma20 = pd.Series(close).rolling(20).mean()
                std20 = pd.Series(close).rolling(20).std()
                upper = ma20 + 2 * std20
                lower = ma20 - 2 * std20
                bb_pos = ((close[-20:] - lower.values[-20:]) / 
                         (upper.values[-20:] - lower.values[-20:] + 1e-8))
                features.extend(bb_pos)
            else:
                features.extend(np.zeros(20))
            
            # 確保總共220維
            features = np.array(features[:220])
            if len(features) < 220:
                features = np.pad(features, (0, 220 - len(features)))
                
        except Exception as e:
            features = np.zeros(220)
        
        return features.astype(np.float32)

class TradingEnvironment:
    """交易環境"""
    def __init__(self, data: Dict, config: PPOConfig):
        self.data = data
        self.config = config
        self.symbols = list(data.keys())
        self.reset()
        
    def reset(self):
        if not self.symbols:
            return np.zeros(self.config.obs_dim)
            
        self.current_symbol = random.choice(self.symbols)
        self.symbol_data = self.data[self.current_symbol]
        
        if len(self.symbol_data) < 100:
            return self.reset()
            
        self.current_step = random.randint(100, min(300, len(self.symbol_data) - 50))
        
        self.cash = self.config.initial_capital
        self.positions = {}
        self.portfolio_value = self.cash
        
        return self._get_observation()
    
    def _get_observation(self):
        try:
            end_idx = self.current_step
            start_idx = max(0, end_idx - 250)
            window_data = self.symbol_data.iloc[start_idx:end_idx]
            
            data_loader = CapitalStockDataLoader()
            features = data_loader.prepare_features(window_data)
            return features
        except:
            return np.zeros(self.config.obs_dim)
    
    def step(self, action):
        try:
            current_price = float(self.symbol_data.iloc[self.current_step]['Close'])
            
            # 執行交易
            if action == 1 or action == 3:  # Buy
                size = 0.1 if action == 1 else 0.2
                if self.cash > 1000:
                    investment = self.cash * size
                    shares = investment / current_price
                    cost = investment * (1 + self.config.transaction_cost)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.positions[self.current_symbol] = \
                            self.positions.get(self.current_symbol, 0) + shares
                        
            elif action == 2:  # Sell
                if self.current_symbol in self.positions:
                    shares = self.positions[self.current_symbol]
                    revenue = shares * current_price * (1 - self.config.transaction_cost)
                    self.cash += revenue
                    del self.positions[self.current_symbol]
            
            self.current_step += 1
            
            # 計算新價值
            new_portfolio_value = self.cash
            for symbol, shares in self.positions.items():
                if symbol == self.current_symbol and self.current_step < len(self.symbol_data):
                    price = float(self.symbol_data.iloc[self.current_step]['Close'])
                    new_portfolio_value += shares * price
            
            # 計算獎勵
            reward = (new_portfolio_value - self.portfolio_value) / (self.portfolio_value + 1e-8)
            self.portfolio_value = new_portfolio_value
            
            # 檢查結束
            done = self.current_step >= min(len(self.symbol_data) - 1, self.current_step + 50)
            
            return self._get_observation(), reward, done
            
        except:
            return np.zeros(self.config.obs_dim), 0, True

class PPOTrainer:
    """PPO訓練器"""
    def __init__(self, config: PPOConfig):
        self.config = config
        self.model = ActorCritic(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        self.training_history = {
            'rewards': [],
            'losses': [],
            'iterations': []
        }
        
    def train(self, env: TradingEnvironment, n_iterations: int = 300):
        """訓練PPO模型"""
        print(f"\n[TRAINING] Starting PPO training for {n_iterations} iterations")
        print(f"[TRAINING] Device: {self.config.device}")
        print(f"[TRAINING] Stocks in environment: {len(env.symbols)}")
        
        for iteration in range(n_iterations):
            # 收集經驗
            batch_obs = []
            batch_acts = []
            batch_rewards = []
            batch_values = []
            batch_log_probs = []
            
            obs = env.reset()
            
            for _ in range(self.config.n_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.config.device)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    action, log_prob, value, _ = self.model.get_action_and_value(obs_tensor)
                
                batch_obs.append(obs)
                batch_acts.append(action.cpu().numpy())
                batch_log_probs.append(log_prob.cpu().numpy())
                batch_values.append(value.cpu().numpy())
                
                obs, reward, done = env.step(action.item())
                batch_rewards.append(reward)
                
                if done:
                    obs = env.reset()
            
            # 計算優勢
            advantages = self._compute_advantages(batch_rewards, batch_values)
            returns = advantages + np.array(batch_values).squeeze()
            
            # PPO更新
            loss = self._ppo_update(
                batch_obs, batch_acts, batch_log_probs, advantages, returns
            )
            
            # 記錄
            avg_reward = np.mean(batch_rewards)
            self.training_history['iterations'].append(iteration)
            self.training_history['rewards'].append(avg_reward)
            self.training_history['losses'].append(loss)
            
            if iteration % 10 == 0:
                print(f"[Iter {iteration:4d}] Reward: {avg_reward:.6f}, Loss: {loss:.4f}")
            
            if iteration % 50 == 0 and iteration > 0:
                self.save_checkpoint(iteration)
    
    def _compute_advantages(self, rewards, values):
        """計算優勢函數"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def _ppo_update(self, observations, actions, old_log_probs, advantages, returns):
        """PPO策略更新"""
        obs_tensor = torch.FloatTensor(observations).to(self.config.device)
        action_tensor = torch.LongTensor(actions).squeeze().to(self.config.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).squeeze().to(self.config.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.config.device)
        returns_tensor = torch.FloatTensor(returns).to(self.config.device)
        
        # 正規化優勢
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.config.n_epochs):
            action_logits, values = self.model(obs_tensor)
            dist = Categorical(logits=action_logits)
            
            new_log_probs = dist.log_prob(action_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # 策略損失
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 價值損失
            value_loss = F.mse_loss(values.squeeze(), returns_tensor)
            
            # 熵損失
            entropy = dist.entropy().mean()
            
            # 總損失
            loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss = loss.item()
        
        return total_loss
    
    def save_checkpoint(self, iteration):
        """保存檢查點"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        filename = f'ppo_capital_{iteration}.pt'
        torch.save(checkpoint, filename)
        print(f"[SAVE] Checkpoint saved: {filename}")

def main():
    print("="*80)
    print("PPO TRAINING WITH CAPITAL.COM STOCKS")
    print("="*80)
    
    # 1. 載入Capital.com股票
    data_loader = CapitalStockDataLoader(use_all=True)  # 使用全部股票
    tickers = data_loader.load_capital_stocks()
    
    if not tickers:
        print("[ERROR] No tickers loaded")
        return
    
    # 2. 下載數據
    stock_data = data_loader.download_stock_data(tickers)
    
    if len(stock_data) < 10:
        print("[ERROR] Insufficient data for training")
        return
    
    # 3. 初始化環境和訓練器
    config = PPOConfig()
    env = TradingEnvironment(stock_data, config)
    trainer = PPOTrainer(config)
    
    # 4. 開始訓練
    print(f"\n[TRAINING] Starting training with {len(stock_data)} Capital.com stocks")
    trainer.train(env, n_iterations=300)
    
    # 5. 保存最終模型
    final_checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'training_history': trainer.training_history,
        'config': config.__dict__,
        'stocks_count': len(stock_data),
        'training_date': datetime.now().isoformat()
    }
    
    torch.save(final_checkpoint, 'ppo_capital_final.pt')
    print(f"\n[COMPLETE] Final model saved: ppo_capital_final.pt")
    
    # 6. 生成報告
    report = {
        'training_date': datetime.now().isoformat(),
        'stocks_count': len(stock_data),
        'total_iterations': 300,
        'final_reward': trainer.training_history['rewards'][-1] if trainer.training_history['rewards'] else 0,
        'avg_reward_last_50': np.mean(trainer.training_history['rewards'][-50:]) if len(trainer.training_history['rewards']) > 50 else 0,
        'device': str(config.device),
        'sample_stocks': list(stock_data.keys())[:20]
    }
    
    with open('ppo_capital_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Stocks used: {len(stock_data)}")
    print(f"Final reward: {report['final_reward']:.6f}")
    print(f"Avg reward (last 50): {report['avg_reward_last_50']:.6f}")
    print("="*80)

if __name__ == "__main__":
    main()