# Cloud DE - Day Trading RL Environment Task

## 任務編號: DT-001
## 優先級: CRITICAL
## 預計時間: 8 小時

## 任務目標
建立強化學習交易環境，符合 OpenAI Gym 標準，支援日內交易策略訓練。

## 具體要求

### 1. 創建 Trading Environment
```python
src/rl_trading/trading_env.py
```

實現功能：
- 繼承 `gym.Env` 基類
- 定義觀察空間 (observation_space)
- 定義動作空間 (action_space)
- 實現核心方法：
  - `reset()`: 重置環境
  - `step()`: 執行動作
  - `render()`: 可視化（可選）

### 2. 狀態空間設計
觀察值應包含：
- 價格數據：OHLCV (5分鐘)
- 技術特徵：
  - 價格變化率
  - 成交量比率
  - RSI/MACD 等指標
- 持倉信息：
  - 當前持倉
  - 未實現盈虧
  - 可用資金

### 3. 動作空間設計
```python
actions = {
    0: "HOLD",     # 持有/不動作
    1: "BUY",      # 買入
    2: "SELL",     # 賣出
    3: "CLOSE"     # 平倉
}
```

### 4. 獎勵函數設計
```python
def calculate_reward(self):
    # 考慮因素：
    # - 實現盈虧
    # - 夏普比率
    # - 最大回撤懲罰
    # - 交易成本
    # - 持倉時間（鼓勵短期交易）
```

### 5. 市場模擬器
```python
src/rl_trading/market_simulator.py
```
- 滑點模擬
- 手續費計算
- 訂單執行延遲
- 部分成交處理

### 6. 數據預處理
```python
src/rl_trading/data_preprocessor.py
```
- 載入分鐘級數據
- 特徵工程
- 數據標準化
- 滾動窗口生成

## 輸出要求

1. **環境類結構**
```python
class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super().__init__()
        # 初始化
    
    def reset(self):
        # 重置環境狀態
        return observation
    
    def step(self, action):
        # 執行交易動作
        return observation, reward, done, info
```

2. **測試腳本**
```python
scripts/test_trading_env.py
```
驗證環境是否正常運作

3. **性能指標**
- 每秒可執行步數 > 1000
- 支援向量化環境
- 內存使用 < 1GB

## 技術要求
- 使用 gymnasium (新版 gym)
- 支援 GPU 加速（如果可用）
- 可並行運行多個環境實例

## 參考資源
- OpenAI Gym documentation
- stable-baselines3 環境要求
- FinRL 框架參考

## 完成標準
- [ ] 環境通過 `gym.utils.env_checker.check_env()` 檢查
- [ ] 可成功運行 1000 個 episode 不崩潰
- [ ] 獎勵函數合理（非恆定值）
- [ ] 支援至少 3 個月的分鐘數據回測

---
**截止時間**: 24小時內
**回報方式**: 創建 PR 並更新進度