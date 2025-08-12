# Cloud Quant - PPO Algorithm Implementation Task

## 任務編號: DT-002
## 優先級: CRITICAL
## 預計時間: 16 小時
## 依賴: DT-001 (Trading Environment)

## 任務目標
實作 PPO (Proximal Policy Optimization) 強化學習算法，訓練日內交易智能體。

## 具體要求

### 1. PPO 算法核心
```python
src/rl_trading/ppo_agent.py
```

實現組件：
- Actor Network (政策網絡)
- Critic Network (價值網絡)
- PPO Loss 計算
- Advantage 估計 (GAE)

### 2. 神經網絡架構
```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

### 3. PPO 訓練循環
```python
class PPOTrainer:
    def __init__(self, 
                 env,
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 epochs=10):
        # 初始化參數
    
    def train(self, num_timesteps):
        # 主訓練循環
        # 1. 收集經驗
        # 2. 計算優勢
        # 3. 更新政策
        # 4. 記錄指標
```

### 4. 經驗收集器
```python
src/rl_trading/rollout_buffer.py
```
- 存儲 trajectories
- 計算 returns 和 advantages
- 批量採樣
- 支援 GPU tensors

### 5. 訓練腳本
```python
scripts/train_ppo_trader.py
```
功能：
- 載入環境
- 配置超參數
- 執行訓練
- 保存檢查點
- TensorBoard 記錄

### 6. 超參數優化
```python
hyperparameters = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'n_steps': [256, 512, 1024],
    'gamma': [0.95, 0.99],
    'gae_lambda': [0.9, 0.95],
    'clip_ratio': [0.1, 0.2, 0.3],
    'hidden_layers': [2, 3],
    'hidden_size': [128, 256, 512]
}
```

### 7. 策略評估
```python
src/rl_trading/evaluator.py
```
評估指標：
- 總收益率
- 夏普比率
- 最大回撤
- 勝率
- 平均持倉時間
- 交易頻率

## 輸出要求

### 1. 核心算法實現
- PPO loss function
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function loss
- Entropy bonus

### 2. 訓練結果
- 收斂曲線 (reward vs episodes)
- 最佳模型權重
- 超參數配置文件
- 訓練日誌

### 3. 回測報告
```python
{
    'total_return': float,
    'sharpe_ratio': float,
    'max_drawdown': float,
    'win_rate': float,
    'avg_trade_return': float,
    'num_trades': int,
    'best_trade': float,
    'worst_trade': float
}
```

## 技術要求

### 必須使用：
- PyTorch 2.0+
- stable-baselines3 (參考實現)
- tensorboard
- wandb (可選)

### 性能要求：
- 訓練速度: > 10k steps/hour
- GPU 利用率: > 80%
- 收斂時間: < 1M timesteps

## 算法細節

### PPO Objective:
```python
L_CLIP = min(
    ratio * advantages,
    clip(ratio, 1-ε, 1+ε) * advantages
)

L_VF = 0.5 * (returns - values)^2

L_S = entropy_coefficient * entropy

Loss = -L_CLIP + c1 * L_VF - c2 * L_S
```

### 訓練流程:
1. Run policy for T timesteps
2. Compute advantage estimates
3. Optimize surrogate L wrt θ (K epochs)
4. θ_old ← θ

## 驗收標準

- [ ] PPO 算法正確實現（通過單元測試）
- [ ] 在測試環境中收斂（reward 持續上升）
- [ ] 回測夏普比率 > 1.5
- [ ] 年化收益 > 20%
- [ ] 最大回撤 < 15%
- [ ] 代碼有完整註釋和文檔

## 參考資源
- Original PPO paper (Schulman et al., 2017)
- OpenAI Spinning Up PPO tutorial
- stable-baselines3 PPO implementation
- FinRL PPO trading examples

---
**截止時間**: 48小時內
**回報方式**: 提交代碼、訓練日誌和回測報告