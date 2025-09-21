# 強化學習交易環境

## 概述

這是一個專為個股當沖（Day Trading）策略設計的強化學習環境，整合了 LSTM 價格預測和 FinBERT 情緒分析作為狀態輸入，提供完整的交易模擬功能。

## 主要特點

### 1. 狀態空間整合
- **市場數據特徵**：OHLCV、技術指標（RSI、MACD、ATR等）
- **LSTM預測特徵**：1日/5日/20日預測值及置信度
- **情緒分析特徵**：FinBERT情緒分數、新聞量、情緒趨勢
- **投資組合狀態**：當前持倉、可用資金、未實現盈虧
- **時間特徵**：交易時段、日內時間、市場階段

### 2. 靈活的動作空間
- **離散動作**：買入（25%/50%/75%/100%倉位）、賣出、持有
- **連續動作**：倉位比例 [-1, 1]（支援做空）
- **動作約束**：考慮資金限制、最大持倉、交易頻率

### 3. 風險調整獎勵函數
```python
reward = profit - risk_penalty - transaction_cost
```
- 考慮波動率懲罰、最大回撤懲罰
- 包含滑價和手續費
- 鼓勵適當的持倉時間

### 4. 風險管理整合
- 倉位大小限制
- 每日虧損限制
- 最大回撤控制
- Kelly準則倉位計算

## 安裝與使用

### 基本使用範例

```python
from rl_trading.environments import TradingEnvironment

# 創建環境
env = TradingEnvironment(
    symbol='AAPL',
    initial_capital=10000,
    max_steps=252,  # 一個交易年
    action_type='discrete'
)

# 重置環境
state = env.reset()

# 執行交易
for _ in range(100):
    action = env.action_space.sample()  # 隨機動作
    state, reward, done, info = env.step(action)
    
    if done:
        break

# 獲取交易摘要
summary = env.get_episode_summary()
print(f"總收益率: {summary['total_return']:.2%}")
print(f"夏普比率: {summary['sharpe_ratio']:.2f}")
```

### 使用預設配置

```python
from rl_trading.configs.env_config import get_config

# 保守策略
config = get_config('conservative')

# 積極策略
config = get_config('aggressive')

# 日內交易
config = get_config('day_trading')

# 波段交易
config = get_config('swing_trading')
```

### 整合 LSTM 和 FinBERT

環境會自動從以下位置載入模型輸出：
- LSTM預測：`src/models/ml_models/`
- 情緒分析：`src/models/sentiment/`

如果實際模型不可用，會使用模擬數據進行測試。

## 環境架構

```
rl_trading/
├── environments/
│   ├── trading_env.py          # 主交易環境
│   ├── state_processor.py      # 狀態空間處理
│   ├── action_space.py         # 動作空間定義
│   └── reward_calculator.py    # 獎勵函數計算
├── utils/
│   ├── portfolio_tracker.py    # 投資組合追蹤
│   └── risk_manager.py         # 風險管理工具
├── configs/
│   └── env_config.py           # 環境配置
└── example_usage.py            # 使用範例
```

## 狀態特徵詳解

### 市場特徵（Market Features）
- `market_close`: 收盤價
- `market_close_change`: 價格變化率
- `market_volume`: 成交量
- `market_range`: 高低價區間

### LSTM特徵（LSTM Features）
- `lstm_pred_1d`: 1日預測價格
- `lstm_return_1d`: 1日預測收益率
- `lstm_conf_1d`: 1日預測置信度
- `lstm_trend_short`: 短期趨勢（5日-1日）
- `lstm_trend_long`: 長期趨勢（20日-5日）

### 情緒特徵（Sentiment Features）
- `sentiment_current`: 當前情緒分數
- `sentiment_mean`: 平均情緒分數
- `sentiment_trend`: 情緒趨勢
- `sentiment_volatility`: 情緒波動性
- `sentiment_news_count`: 新聞數量

### 投資組合特徵（Portfolio Features）
- `portfolio_position`: 當前持倉
- `portfolio_position_pct`: 持倉占比
- `portfolio_unrealized_pnl`: 未實現盈虧
- `portfolio_cash`: 可用現金
- `portfolio_exposure`: 風險暴露度

## 動作空間詳解

### 離散動作（預設）
- `HOLD` (0): 持有不動
- `BUY_25` (1): 買入25%可用資金
- `BUY_50` (2): 買入50%可用資金
- `BUY_75` (3): 買入75%可用資金
- `BUY_100` (4): 買入100%可用資金
- `SELL_25` (5): 賣出25%持倉
- `SELL_50` (6): 賣出50%持倉
- `SELL_75` (7): 賣出75%持倉
- `SELL_100` (8): 清空持倉

### 連續動作
- 範圍：[-1, 1]
- 正值表示做多，負值表示做空（如果允許）
- 絕對值表示倉位大小

## 獎勵函數組成

1. **基礎收益**：價格變動帶來的盈虧
2. **交易成本**：手續費和滑價
3. **風險懲罰**：
   - 波動率懲罰
   - 回撤懲罰
   - 過度暴露懲罰
4. **行為獎勵**：
   - 持有獲利倉位的獎勵
   - 過度交易的懲罰
   - 市場時機把握獎勵

## 性能指標

環境會追蹤以下關鍵指標：
- **總收益率**：投資組合最終價值變化
- **夏普比率**：風險調整後收益
- **最大回撤**：最大虧損幅度
- **勝率**：獲利交易比例
- **平均交易規模**：典型交易大小

## 進階使用

### 自定義狀態處理器

```python
from rl_trading.environments.state_processor import StateConfig

config = StateConfig(
    price_features=['close', 'high', 'low'],
    lstm_horizons=[1, 5, 10],
    sentiment_window=48,  # 48小時
    include_time_features=True
)
```

### 自定義獎勵函數

```python
from rl_trading.environments.reward_calculator import RewardConfig

reward_config = RewardConfig(
    use_risk_adjusted_returns=True,
    volatility_penalty=0.2,
    max_drawdown_penalty=0.3,
    commission_rate=0.001
)
```

### 風險管理設定

```python
from rl_trading.utils.risk_manager import RiskLimits

risk_limits = RiskLimits(
    max_position_size=0.25,     # 單一持倉不超過25%
    max_daily_loss=0.02,        # 每日最大虧損2%
    max_drawdown=0.10,          # 最大回撤10%
    position_size_kelly=True    # 使用Kelly準則
)
```

## 與強化學習庫整合

### Stable Baselines3 範例

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 創建向量化環境
def make_env():
    return TradingEnvironment(symbol='AAPL')

vec_env = DummyVecEnv([make_env for _ in range(4)])

# 訓練PPO模型
model = PPO('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=100000)
```

## 注意事項

1. **數據需求**：環境需要足夠的歷史數據（至少 window_size + max_steps）
2. **計算資源**：整合LSTM和FinBERT會增加計算需求
3. **交易限制**：遵守實際市場的交易規則和限制
4. **風險控制**：始終設置適當的風險限制

## 故障排除

### 常見問題

1. **ImportError**：確保所有依賴模組已正確安裝
2. **數據不足**：增加歷史數據或減少 window_size
3. **獎勵爆炸**：調整 reward_scaling 或啟用 clip_reward

### 調試建議

- 使用 `render_mode='human'` 查看詳細交易信息
- 檢查 `info` 字典中的診斷信息
- 使用較小的 `max_steps` 進行快速測試

## 未來改進

- [ ] 支援多資產交易
- [ ] 整合更多技術指標
- [ ] 添加市場微結構特徵
- [ ] 實現更複雜的獎勵函數
- [ ] 支援期權交易策略