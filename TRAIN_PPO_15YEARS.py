#!/usr/bin/env python3
"""
PPO訓練 - 使用Capital.com 4445個股票 + 15年歷史數據
智能緩存系統：重用已下載的數據
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
import pickle
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ppo_15years_training.log'),
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

class SmartDataLoader:
    """智能數據加載器 - 支援緩存和15年數據"""
    def __init__(self):
        self.data_cache = {}
        self.cache_dir = "data_cache"
        self.start_date = "2010-01-01"  # 15年歷史數據！
        
        # 創建緩存目錄
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"[CACHE] Created cache directory: {self.cache_dir}")
    
    def load_capital_stocks(self):
        """載入Capital.com股票列表"""
        if os.path.exists('capital_real_tickers.txt'):
            with open('capital_real_tickers.txt', 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            print(f"[DATA] Loaded {len(tickers)} Capital.com tickers")
            return tickers
        else:
            print("[ERROR] capital_real_tickers.txt not found")
            return []
    
    def get_cache_filename(self, ticker: str) -> str:
        """獲取緩存文件名"""
        return os.path.join(self.cache_dir, f"{ticker}_15years.pkl")
    
    def load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """從緩存載入數據"""
        cache_file = self.get_cache_filename(ticker)
        if os.path.exists(cache_file):
            try:
                # 檢查緩存是否過期（超過7天）
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                if file_age.days < 7:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    return data
            except:
                pass
        return None
    
    def save_to_cache(self, ticker: str, data: pd.DataFrame):
        """保存數據到緩存"""
        cache_file = self.get_cache_filename(ticker)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def download_stock_data(self, tickers: List[str]):
        """智能下載股票數據（使用緩存）"""
        print(f"[DATA] Processing {len(tickers)} stocks...")
        print(f"[DATA] Date range: {self.start_date} to {datetime.now().date()} (15 YEARS!)")
        print(f"[CACHE] Checking cache for existing data...")
        
        cached_count = 0
        download_needed = []
        
        # 檢查緩存
        for ticker in tickers:
            cached_data = self.load_from_cache(ticker)
            if cached_data is not None and len(cached_data) > 1000:  # 至少4年數據
                self.data_cache[ticker] = cached_data
                cached_count += 1
            else:
                download_needed.append(ticker)
        
        print(f"[CACHE] Found {cached_count} stocks in cache")
        print(f"[DOWNLOAD] Need to download {len(download_needed)} stocks")
        
        if download_needed:
            # 批量下載
            success_count = 0
            failed_tickers = []
            
            # 分批下載
            batch_size = 20
            batches = [download_needed[i:i+batch_size] for i in range(0, len(download_needed), batch_size)]
            
            for batch in tqdm(batches, desc="Downloading new data"):
                for ticker in batch:
                    try:
                        # 下載15年數據
                        data = yf.download(
                            ticker,
                            start=self.start_date,
                            end=datetime.now().strftime('%Y-%m-%d'),
                            interval='1d',
                            progress=False,
                            threads=False
                        )
                        
                        if not data.empty and len(data) > 1000:  # 至少4年數據
                            self.data_cache[ticker] = data
                            self.save_to_cache(ticker, data)  # 保存到緩存
                            success_count += 1
                        else:
                            failed_tickers.append(ticker)
                            
                    except Exception as e:
                        failed_tickers.append(ticker)
                
                # 每批次後顯示進度
                if success_count > 0 and success_count % 100 == 0:
                    print(f"[PROGRESS] Downloaded {success_count} new stocks...")
            
            print(f"[DOWNLOAD] Successfully downloaded: {success_count}/{len(download_needed)} stocks")
            if failed_tickers[:10]:
                print(f"[FAILED] Sample failed tickers: {failed_tickers[:10]}")
        
        print(f"\n[TOTAL] Total stocks ready for training: {len(self.data_cache)}")
        return self.data_cache
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵（使用15年數據的豐富特徵）"""
        features = []
        
        try:
            # 1. 短期收益率 (50維)
            returns = data['Close'].pct_change().fillna(0).values[-50:]
            features.extend(returns if len(returns) == 50 else np.zeros(50))
            
            # 2. 長期收益率 (20維) - 利用15年數據
            if len(data) > 252:
                yearly_returns = []
                for days in [20, 60, 120, 252]:  # 1月, 3月, 6月, 1年
                    if len(data) > days:
                        ret = (data['Close'].iloc[-1] / data['Close'].iloc[-days] - 1)
                        yearly_returns.append(ret)
                    else:
                        yearly_returns.append(0)
                features.extend(yearly_returns * 5)  # 重複5次得到20維
            else:
                features.extend(np.zeros(20))
            
            # 3. 成交量特徵 (20維)
            if 'Volume' in data.columns:
                volume = data['Volume'].values[-20:]
                volume_norm = volume / (volume.mean() + 1e-8)
                features.extend(volume_norm if len(volume_norm) == 20 else np.zeros(20))
            else:
                features.extend(np.zeros(20))
            
            # 4. RSI多週期 (30維)
            close = data['Close'].values
            for period in [7, 14, 21]:  # 多週期RSI
                if len(close) >= period:
                    delta = pd.Series(close).diff()
                    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                    rs = gain / (loss + 1e-8)
                    rsi = 100 - (100 / (1 + rs))
                    features.extend(rsi.fillna(50).values[-10:])
                else:
                    features.extend(np.zeros(10))
            
            # 5. 移動平均 (40維)
            for period in [5, 10, 20, 50]:
                if len(close) >= period:
                    sma = pd.Series(close).rolling(period).mean().values[-10:]
                    features.extend(sma if len(sma) == 10 else np.zeros(10))
                else:
                    features.extend(np.zeros(10))
            
            # 6. MACD (30維)
            if len(close) >= 26:
                exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
                exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                features.extend(macd.values[-30:])
            else:
                features.extend(np.zeros(30))
            
            # 7. 布林帶 (20維)
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
            
            # 8. 歷史波動率 (10維) - 利用15年數據
            if len(data) > 252:
                for window in [20, 60, 120, 252, 504]:  # 不同時間窗口
                    if len(data) > window:
                        vol = data['Close'].pct_change().rolling(window).std()
                        features.extend([vol.iloc[-1], vol.iloc[-1]])
                    else:
                        features.extend([0, 0])
            else:
                features.extend(np.zeros(10))
            
            # 確保總共220維
            features = np.array(features[:220])
            if len(features) < 220:
                features = np.pad(features, (0, 220 - len(features)))
                
        except Exception as e:
            features = np.zeros(220)
        
        return features.astype(np.float32)

class TradingEnvironment:
    """交易環境 - 支援15年數據"""
    def __init__(self, data: Dict, config: PPOConfig):
        self.data = data
        self.config = config
        self.symbols = list(data.keys())
        self.reset()
        
    def reset(self):
        if not self.symbols:
            return np.zeros(self.config.obs_dim)
            
        self.current_symbol = np.random.choice(self.symbols)
        self.symbol_data = self.data[self.current_symbol]
        
        # 使用更長的歷史窗口（因為有15年數據）
        min_history = 252  # 至少1年
        if len(self.symbol_data) < min_history:
            return self.reset()
            
        # 在更大範圍內隨機選擇起始點
        max_start = len(self.symbol_data) - 100
        self.current_step = np.random.randint(min_history, max_start)
        
        self.cash = self.config.initial_capital
        self.positions = {}
        self.portfolio_value = self.cash
        
        return self._get_observation()
    
    def _get_observation(self):
        try:
            end_idx = self.current_step
            # 使用更長的歷史窗口（最多1000天）
            start_idx = max(0, end_idx - 1000)
            window_data = self.symbol_data.iloc[start_idx:end_idx]
            
            data_loader = SmartDataLoader()
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
            done = self.current_step >= len(self.symbol_data) - 1
            
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
        
    def train(self, env: TradingEnvironment, n_iterations: int = 500):
        """訓練PPO模型"""
        print(f"\n[TRAINING] Starting PPO training for {n_iterations} iterations")
        print(f"[TRAINING] Device: {self.config.device}")
        print(f"[TRAINING] Stocks in environment: {len(env.symbols)}")
        print(f"[TRAINING] Using 15 YEARS of historical data!")
        
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
        
        filename = f'ppo_15years_{iteration}.pt'
        torch.save(checkpoint, filename)
        print(f"[SAVE] Checkpoint saved: {filename}")

def main():
    print("="*80)
    print("PPO TRAINING WITH 15 YEARS OF DATA")
    print("USING ALL 4445 CAPITAL.COM STOCKS")
    print("="*80)
    
    # 1. 載入Capital.com股票
    data_loader = SmartDataLoader()
    tickers = data_loader.load_capital_stocks()
    
    if not tickers:
        print("[ERROR] No tickers loaded")
        return
    
    # 2. 智能下載數據（使用緩存）
    stock_data = data_loader.download_stock_data(tickers)
    
    if len(stock_data) < 100:
        print("[WARNING] Less than 100 stocks available, but continuing...")
    
    # 3. 初始化環境和訓練器
    config = PPOConfig()
    env = TradingEnvironment(stock_data, config)
    trainer = PPOTrainer(config)
    
    # 4. 開始訓練
    print(f"\n[TRAINING] Starting training with {len(stock_data)} stocks")
    print(f"[TRAINING] Each stock has up to 15 YEARS of data!")
    trainer.train(env, n_iterations=500)
    
    # 5. 保存最終模型
    final_checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'training_history': trainer.training_history,
        'config': config.__dict__,
        'stocks_count': len(stock_data),
        'training_date': datetime.now().isoformat(),
        'data_range': '2010-2025 (15 YEARS)'
    }
    
    torch.save(final_checkpoint, 'ppo_15years_final.pt')
    print(f"\n[COMPLETE] Final model saved: ppo_15years_final.pt")
    
    # 6. 生成報告
    report = {
        'training_date': datetime.now().isoformat(),
        'stocks_count': len(stock_data),
        'total_iterations': 500,
        'data_years': 15,
        'final_reward': trainer.training_history['rewards'][-1] if trainer.training_history['rewards'] else 0,
        'avg_reward_last_50': np.mean(trainer.training_history['rewards'][-50:]) if len(trainer.training_history['rewards']) > 50 else 0,
        'device': str(config.device),
        'sample_stocks': list(stock_data.keys())[:20]
    }
    
    with open('ppo_15years_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Stocks used: {len(stock_data)}")
    print(f"Data range: 15 YEARS (2010-2025)")
    print(f"Final reward: {report['final_reward']:.6f}")
    print("="*80)

if __name__ == "__main__":
    main()