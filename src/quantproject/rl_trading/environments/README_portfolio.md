# 多資產投資組合管理環境

## 概述

本模組實現了支援多資產投資組合管理的強化學習環境，從原本的單股票交易擴展到多股票投資組合優化。

## 主要特性

### 1. 狀態空間設計
- **多資產特徵**：每個資產包含 20 個技術和基本面特徵
- **投資組合特徵**：10 個組合層級的特徵（現金比例、集中度、收益統計等）
- **總維度**：`n_assets * 20 + 10`

### 2. 動作空間設計
- **連續權重分配**：為每個資產（包含現金）分配 0-1 的權重
- **自動正規化**：確保所有權重總和為 1
- **支援做多策略**：目前版本只支援做多（權重 >= 0）

### 3. 獎勵函數
- **風險調整收益**：基於 Sharpe Ratio 的獎勵
- **最大回撤懲罰**：避免過大的資金回撤
- **分散化獎勵**：鼓勵投資組合分散化
- **交易成本考量**：懲罰過度交易

### 4. 再平衡機制
- **定期再平衡**：預設每 5 個交易日允許再平衡
- **智能執行**：考慮滑價和交易成本
- **部位限制**：確保不會超買或超賣

## 使用範例

### 基本使用

```python
from rl_trading.environments.portfolio_env import PortfolioTradingEnvironment
from rl_trading.agents.portfolio_agent import PortfolioAgent

# 創建環境
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
env = PortfolioTradingEnvironment(
    symbols=symbols,
    initial_capital=100000,
    transaction_cost=0.001,
    slippage=0.0005,
    rebalance_frequency=5
)

# 創建代理
agent = PortfolioAgent(env, n_assets=len(symbols))

# 訓練
agent.train(total_timesteps=1000000)
```

### 進階配置

```python
# 保守策略
conservative_env = PortfolioTradingEnvironment(
    symbols=symbols,
    transaction_cost=0.002,  # 較高的交易成本
    rebalance_frequency=10,  # 較少的再平衡
    risk_free_rate=0.03     # 較高的無風險利率要求
)

# 積極策略
aggressive_env = PortfolioTradingEnvironment(
    symbols=symbols,
    transaction_cost=0.0005,  # 較低的交易成本
    rebalance_frequency=3,    # 頻繁再平衡
    risk_free_rate=0.01      # 較低的無風險利率要求
)
```

## 訓練指令

### 基本訓練

```bash
python train_portfolio.py --symbols AAPL GOOGL MSFT AMZN TSLA
```

### 自定義訓練

```bash
python train_portfolio.py \
    --symbols AAPL GOOGL MSFT NVDA TSLA META \
    --strategy conservative \
    --initial-capital 1000000 \
    --rebalance-freq 10 \
    --total-timesteps 2000000 \
    --output-dir ./models/portfolio_conservative
```

## 性能指標

### 關鍵指標
- **總回報率**：投資期間的總收益
- **年化回報率**：年化後的收益率
- **Sharpe Ratio**：風險調整後收益
- **最大回撤**：最大資金下降幅度
- **勝率**：正收益天數比例
- **平均交易成本**：每筆交易的平均成本

### 評估方法

```python
# 評估訓練好的代理
metrics = agent.evaluate(
    eval_env=test_env,
    n_episodes=50,
    deterministic=True
)

print(f"平均年化收益: {metrics['mean_return'] * 252:.2%}")
print(f"Sharpe Ratio: {metrics['mean_sharpe']:.2f}")
print(f"最大回撤: {metrics['mean_max_drawdown']:.2%}")
```

## 技術細節

### 特徵工程
每個資產的特徵包括：
- 收益率統計（均值、標準差、偏度、峰度）
- 技術指標（RSI、MACD、移動平均）
- 價格位置（相對高低點）
- 成交量特徵
- 動量指標

### 投資組合特徵
- 現金比例
- 集中度（Herfindahl 指數）
- 活躍部位數量
- 歷史收益統計
- 距離上次再平衡天數
- 相對初始資金的收益

### 網絡架構
- **特徵提取器**：每個資產獨立編碼
- **注意力機制**：捕捉資產間關係
- **組合層**：整合資產和組合特徵
- **策略網絡**：輸出權重分配
- **價值網絡**：估計狀態價值

## 最佳實踐

### 1. 資產選擇
- 選擇流動性高的資產
- 確保資產間有一定相關性但不完全相關
- 考慮不同產業和風格的平衡

### 2. 參數調整
- **再平衡頻率**：5-20 天較為合理
- **交易成本**：根據實際券商費率設定
- **滑價**：大盤股 0.05%，小盤股 0.1-0.2%

### 3. 風險管理
- 設置最大單一資產權重限制
- 監控投資組合集中度
- 定期檢查最大回撤

## 未來改進

1. **空頭支援**：允許做空操作
2. **槓桿交易**：支援保證金交易
3. **更多約束**：如產業限制、ESG 約束
4. **多期優化**：考慮稅務影響
5. **市場體制識別**：根據市場狀態調整策略