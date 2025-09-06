#!/usr/bin/env python3
"""
簡化版PPO訓練器 - 用於處理3488個股票
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from datetime import datetime
import os
import json
from tqdm import tqdm

class PPONetwork(nn.Module):
    """PPO神經網絡"""
    def __init__(self, input_dim=220, hidden_dim=256, output_dim=3):
        super().__init__()
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 共享層 - Reduced complexity
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Actor頭
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Critic頭
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Apply initialization
        self.shared.apply(init_weights)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
    
    def forward(self, x):
        # Ensure input is valid
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        shared = self.shared(x)
        
        # Get action logits and apply softmax
        action_logits = self.actor(shared)
        action_logits = torch.clamp(action_logits, min=-10, max=10)  # Prevent overflow
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Add small epsilon to prevent log(0)
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        value = self.critic(shared)
        value = torch.clamp(value, min=-100, max=100)  # Prevent extreme values
        
        return action_probs, value

class SimplePPOTrainer:
    """簡化的PPO訓練器"""
    def __init__(self, stock_data, config=None):
        self.stock_data = stock_data
        self.stock_list = list(stock_data.keys())
        
        # 默認配置
        self.config = config or {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_range': 0.2,
            'epochs': 10,
            'batch_size': 64,
            'feature_dim': 220,
            'action_dim': 3  # 買入、持有、賣出
        }
        
        # 初始化網絡
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = PPONetwork(
            input_dim=self.config['feature_dim'],
            output_dim=self.config['action_dim']
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config['learning_rate'])
        
        # 訓練統計
        self.episode_rewards = []
        self.losses = []
        
    def extract_features(self, stock_symbol, date_idx, window=30):
        """提取220維特徵"""
        try:
            df = self.stock_data[stock_symbol]
            
            if date_idx < window or date_idx >= len(df):
                return np.zeros(220, dtype=np.float32)
            
            # 獲取窗口數據
            window_data = df.iloc[date_idx-window:date_idx]
            
            # Check for valid data
            if window_data.empty or 'Close' not in window_data.columns:
                return np.zeros(220, dtype=np.float32)
            
            features = []
            
            # 1. 價格特徵 (40維)
            prices = window_data['Close'].values
            features.extend([
                prices[-1] / prices[0] - 1,  # 總收益率
                np.mean(prices) / prices[-1] - 1,  # 均值回歸
                np.std(prices) / np.mean(prices),  # 波動率
                (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8),  # 位置
            ])
            
            # 移動平均
            for period in [5, 10, 20, 30]:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    features.append(prices[-1] / ma - 1)
                else:
                    features.append(0)
            
            # 2. 成交量特徵 (30維)
            volumes = window_data['Volume'].values
            features.extend([
                volumes[-1] / (np.mean(volumes) + 1e-8) - 1,
                np.std(volumes) / (np.mean(volumes) + 1e-8),
            ])
            
            # 3. 技術指標 (50維)
            # RSI
            price_diff = np.diff(prices)
            gains = price_diff[price_diff > 0]
            losses = -price_diff[price_diff < 0]
            
            if len(gains) > 0 and len(losses) > 0:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100)
            else:
                features.append(0.5)
            
            # MACD
            if len(prices) >= 26:
                ema12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
                ema26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
                macd = (ema12 - ema26) / prices[-1]
                features.append(macd)
            else:
                features.append(0)
            
            # 補充到220維
            while len(features) < 220:
                features.append(0)
            
            # Convert to numpy array and handle NaN values
            features_array = np.array(features[:220], dtype=np.float32)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize features to prevent exploding gradients
            mean = np.mean(features_array)
            std = np.std(features_array)
            if std > 0:
                features_array = (features_array - mean) / (std + 1e-8)
            
            return features_array
            
        except Exception as e:
            return np.zeros(220, dtype=np.float32)
    
    def train_episode(self, num_steps=1000):
        """訓練一個episode"""
        # 隨機選擇股票
        selected_stocks = random.sample(self.stock_list, min(50, len(self.stock_list)))
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        # 初始化
        balance = 100000
        positions = {}
        
        for step in range(num_steps):
            # 隨機選擇一個股票和時間點
            stock = random.choice(selected_stocks)
            df = self.stock_data[stock]
            
            if len(df) < 100:
                continue
            
            date_idx = random.randint(50, len(df) - 50)
            
            # 提取特徵
            state = self.extract_features(stock, date_idx)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 獲取動作和價值
            with torch.no_grad():
                action_probs, value = self.network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # 執行動作
            current_price = df.iloc[date_idx]['Close']
            next_price = df.iloc[date_idx + 1]['Close'] if date_idx + 1 < len(df) else current_price
            
            reward = 0
            if action == 0:  # 買入
                if balance > current_price * 100:
                    shares = 100
                    balance -= current_price * shares * 1.001  # 手續費
                    positions[stock] = positions.get(stock, 0) + shares
                    reward = (next_price - current_price) * shares / 100000
            elif action == 2:  # 賣出
                if stock in positions and positions[stock] > 0:
                    shares = min(100, positions[stock])
                    balance += current_price * shares * 0.999  # 手續費
                    positions[stock] -= shares
                    reward = shares * current_price / 100000 - 1
            else:  # 持有
                if stock in positions and positions[stock] > 0:
                    reward = (next_price - current_price) * positions[stock] / 100000
            
            # 保存數據
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
        
        # 計算優勢和回報
        returns = []
        advantages = []
        
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            discounted_reward = rewards[i] + self.config['gamma'] * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        for _ in range(self.config['epochs']):
            action_probs, values = self.network(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 計算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 計算損失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_range'], 1 + self.config['clip_range']) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
            
            # 更新網絡
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            self.losses.append(total_loss.item())
        
        # 計算總回報
        total_return = sum(rewards)
        self.episode_rewards.append(total_return)
        
        return total_return

def train_ppo_simple(stock_data):
    """簡化的訓練函數"""
    print("\n開始簡化版PPO訓練...")
    print(f"使用 {len(stock_data)} 個股票的數據")
    
    # 需要pandas
    global pd
    import pandas as pd
    
    # 創建訓練器
    trainer = SimplePPOTrainer(stock_data)
    
    # 訓練參數
    num_episodes = 1000
    save_freq = 100
    
    # 創建保存目錄
    os.makedirs('models/ppo_simple', exist_ok=True)
    os.makedirs('logs/ppo_simple', exist_ok=True)
    
    best_reward = -float('inf')
    
    print(f"開始訓練 {num_episodes} 個episodes...")
    
    for episode in tqdm(range(num_episodes), desc="訓練進度"):
        # 訓練一個episode
        episode_reward = trainer.train_episode()
        
        # 打印進度
        if episode % 10 == 0:
            avg_reward = np.mean(trainer.episode_rewards[-100:]) if len(trainer.episode_rewards) >= 100 else np.mean(trainer.episode_rewards)
            print(f"\nEpisode {episode}, Avg Reward: {avg_reward:.4f}")
        
        # 保存模型
        if episode % save_freq == 0 and episode > 0:
            model_path = f'models/ppo_simple/model_episode_{episode}.pt'
            torch.save({
                'episode': episode,
                'model_state_dict': trainer.network.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'episode_rewards': trainer.episode_rewards,
                'config': trainer.config
            }, model_path)
            
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': trainer.network.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'episode_rewards': trainer.episode_rewards,
                    'config': trainer.config
                }, 'models/ppo_simple/best_model.pt')
                print(f"保存最佳模型，獎勵: {best_reward:.4f}")
    
    # 保存訓練日誌
    log_data = {
        'total_episodes': num_episodes,
        'final_avg_reward': np.mean(trainer.episode_rewards[-100:]),
        'best_reward': best_reward,
        'episode_rewards': trainer.episode_rewards,
        'config': trainer.config,
        'num_stocks': len(stock_data),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('logs/ppo_simple/training_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print("\n" + "="*80)
    print("訓練完成！")
    print(f"最終平均獎勵: {log_data['final_avg_reward']:.4f}")
    print(f"最佳獎勵: {best_reward:.4f}")
    print(f"模型保存在: models/ppo_simple/")
    print("="*80)

if __name__ == "__main__":
    # 測試
    print("簡化版PPO訓練器已準備就緒")