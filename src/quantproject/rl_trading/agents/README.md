# RL Trading Agents

## 概述

本模組實現了用於量化交易的強化學習代理，目前支援 PPO (Proximal Policy Optimization) 算法，專門針對個股當沖策略進行優化。

## 主要特點

### 1. PPO Agent 實現
- **自適應學習率**：動態調整學習速度
- **並行環境訓練**：支援多環境加速訓練
- **自定義網絡架構**：針對交易特徵優化的神經網絡
- **穩定訓練**：使用 PPO 的 clip 機制確保穩定更新

### 2. 神經網絡架構
```
特徵提取器 (256維):
├── 市場特徵
├── LSTM預測特徵
├── 情緒分析特徵
└── 投資組合狀態

Actor網絡:
Input → Dense(256) → Dense(256) → Dense(128) → Actions

Critic網絡:
Input → Dense(256) → Dense(256) → Dense(128) → Value
```

### 3. 訓練功能
- **自動評估**：定期在驗證集上評估
- **早停機制**：防止過擬合
- **檢查點保存**：定期保存模型
- **TensorBoard整合**：實時監控訓練

## 快速開始

### 基本訓練

```bash
python train_agent.py --symbol AAPL --total-timesteps 1000000
```

### 進階設定

```bash
python train_agent.py \
    --symbol AAPL \
    --env-config day_trading \
    --learning-rate 3e-4 \
    --n-envs 8 \
    --total-timesteps 2000000 \
    --eval-freq 10000 \
    --experiment-name "AAPL_PPO_experiment"
```

### 使用不同預設配置

```bash
# 保守策略
python train_agent.py --env-config conservative

# 積極策略
python train_agent.py --env-config aggressive

# 日內交易
python train_agent.py --env-config day_trading

# 波段交易
python train_agent.py --env-config swing_trading
```

## 訓練參數說明

### 環境參數
- `--symbol`: 交易標的（預設: AAPL）
- `--initial-capital`: 初始資金（預設: 10000）
- `--episode-length`: 每回合最大步數（預設: 252）

### PPO超參數
- `--learning-rate`: 學習率（預設: 3e-4）
- `--n-steps`: 每次更新的步數（預設: 2048）
- `--batch-size`: 批次大小（預設: 64）
- `--n-epochs`: 每次更新的訓練輪數（預設: 10）
- `--gamma`: 折扣因子（預設: 0.99）
- `--clip-range`: PPO裁剪範圍（預設: 0.2）

### 訓練設定
- `--total-timesteps`: 總訓練步數（預設: 1000000）
- `--eval-freq`: 評估頻率（預設: 10000）
- `--save-freq`: 模型保存頻率（預設: 50000）
- `--n-eval-episodes`: 評估回合數（預設: 5）

## 訓練監控

### TensorBoard

```bash
tensorboard --logdir results/[experiment_name]/logs/tensorboard
```

### 訓練指標
- **episode_reward**: 每回合總獎勵
- **episode_length**: 回合長度
- **trading/episode_return**: 投資報酬率
- **trading/n_trades**: 交易次數
- **trading/sharpe_ratio**: 夏普比率

## 模型評估

訓練完成後，會自動生成以下評估報告：

1. **訓練報告** (`training_report.txt`)
   - 實驗配置
   - 訓練指標
   - 最佳驗證分數

2. **評估結果** (`evaluation_results.json`)
   - 平均收益率
   - 夏普比率
   - 最大回撤
   - 勝率統計

3. **視覺化圖表**
   - 訓練曲線 (`training_rewards.png`)
   - 投資組合指標 (`portfolio_metrics.png`)
   - 交易行為分析 (`trading_behavior.png`)

## 模型使用

### 載入訓練好的模型

```python
from rl_trading.agents import PPOAgent
from rl_trading.environments import TradingEnvironment

# 創建環境
env = TradingEnvironment(symbol='AAPL')

# 載入代理
agent = PPOAgent(env)
agent.load('results/experiment_name/models/best_model')

# 進行預測
obs = env.reset()
done = False

while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

### 批量評估

```python
from rl_trading.training import Evaluator

evaluator = Evaluator(agent)
results = evaluator.evaluate(
    env=test_env,
    n_episodes=50,
    detailed_analysis=True
)
```

## 超參數調優

### 使用 Optuna 進行超參數搜索

```python
from rl_trading.training import Trainer

trainer = Trainer(agent_type='PPO')

param_space = {
    'learning_rate': (1e-5, 1e-3),
    'n_steps': [1024, 2048, 4096],
    'batch_size': [32, 64, 128],
    'ent_coef': (0.0, 0.1)
}

best_params = trainer.run_hyperparameter_search(
    param_space=param_space,
    n_trials=20,
    n_timesteps_per_trial=100000
)
```

## 性能優化建議

1. **環境並行化**
   - 增加 `n_envs` 可顯著加速訓練
   - 建議使用 CPU 核心數的一半

2. **網絡架構**
   - 對於複雜市場，可增加隱藏層大小
   - 考慮使用 LSTM 處理時序特徵

3. **獎勵函數調整**
   - 根據交易風格調整風險懲罰
   - 平衡收益和交易頻率

4. **訓練穩定性**
   - 使用較小的學習率開始
   - 監控 KL 散度避免更新過大

## 常見問題

### 訓練不收斂
- 降低學習率
- 增加 batch size
- 檢查環境獎勵縮放

### 過度交易
- 增加交易成本
- 調整 entropy coefficient
- 使用 holding reward

### 性能不佳
- 增加訓練時間
- 嘗試不同的網絡架構
- 優化特徵工程

## 未來改進

- [ ] 實現 SAC (Soft Actor-Critic) 算法
- [ ] 添加 Rainbow DQN
- [ ] 支援多資產組合
- [ ] 整合更多市場微結構特徵
- [ ] 實現分層強化學習