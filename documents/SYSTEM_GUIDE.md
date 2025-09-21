# 智能量化交易系統操作指南

## 系統架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                     視覺化儀表板 (Dashboard)                    │
├─────────────────────────────────────────────────────────────┤
│                    主控制器 (Main Controller)                  │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  數據獲取     │   感官模型    │   RL決策引擎  │   風險管理     │
│ Capital.com  │  LSTM/FinBERT│  PPO Agent   │  Position Mgmt │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

## 快速開始

### 1. 環境設置

```bash
# 1. 克隆專案
git clone https://github.com/Shelby-Wong0312/QuantProject.git
cd QuantProject

# 2. 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 設置環境變數
cp .env.example .env
# 編輯 .env 填入 Capital.com API 認證資訊
```

### 2. 系統運行模式

## 模式一：完整系統運行（推薦）

```python
# 運行整合系統
python src/integration/run_system.py

# 或使用配置文件
python src/integration/run_system.py --config config/production.yaml
```

這將啟動：
- 數據收集與處理
- 感官模型推理
- RL Agent 決策
- 自動交易執行
- 實時監控儀表板

## 模式二：訓練模式

### A. 訓練單股票策略
```python
# 1. 準備數據
python src/data_providers/capital_com/example_usage.py --download-history

# 2. 訓練 LSTM 模型
python src/models/ml_models/example_usage.py --train

# 3. 訓練 RL Agent
python src/rl_trading/train_agent.py \
    --symbol AAPL \
    --total-timesteps 1000000 \
    --save-path ./models/rl_agent
```

### B. 訓練投資組合策略
```python
# 訓練多資產投資組合
python src/rl_trading/train_portfolio.py \
    --symbols AAPL GOOGL MSFT AMZN TSLA \
    --strategy balanced \
    --total-timesteps 2000000
```

## 模式三：回測模式

```python
# 運行回測
python src/backtesting/example_usage.py \
    --strategy rl_strategy \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --initial-capital 100000
```

## 模式四：視覺化監控

```python
# 啟動儀表板
python src/visualization/dashboard/app.py

# 瀏覽器訪問: http://localhost:8050
```

儀表板功能：
- **Alpha生成**：查看 K線圖、LSTM 預測、情緒分析
- **投資組合分析**：監控倉位、風險指標、相關性網絡
- **執行效率**：分析滑價、成交分布、決策熱力圖

## 系統配置

### 基本配置文件結構
```yaml
# config/production.yaml
system:
  mode: "live"  # live/paper/backtest
  
data:
  provider: "capital_com"
  symbols: ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
  update_frequency: 60  # seconds

models:
  lstm:
    enabled: true
    model_path: "./models/lstm/best_model.h5"
  
  finbert:
    enabled: true
    update_interval: 300  # seconds
  
  gnn:
    enabled: true
    correlation_threshold: 0.3

trading:
  initial_capital: 100000
  position_size: 0.1  # 10% per trade
  max_positions: 5
  stop_loss: 0.02  # 2%
  take_profit: 0.05  # 5%

rl_agent:
  model_path: "./models/rl/ppo_agent.zip"
  action_space: "discrete"  # discrete/continuous
  update_frequency: "daily"
```

## 核心操作流程

### 1. 數據流程
```python
# 數據獲取 → 清洗 → 特徵工程 → 模型輸入
from quantproject.data_processing.pipeline import DataPipeline

pipeline = DataPipeline()
processed_data = pipeline.process(raw_data)
```

### 2. 模型推理
```python
# LSTM 預測
from quantproject.models.ml_models.lstm_predictor import LSTMPredictor

predictor = LSTMPredictor.load("./models/lstm/")
predictions = predictor.predict(data, horizons=[1, 5, 20])

# 情緒分析
from quantproject.models.sentiment.finbert_analyzer import FinBERTAnalyzer

analyzer = FinBERTAnalyzer()
sentiment_scores = analyzer.analyze(news_data)
```

### 3. RL 決策
```python
# 獲取交易決策
from quantproject.rl_trading.agents.ppo_agent import PPOAgent

agent = PPOAgent.load("./models/rl/")
state = env.get_observation()
action = agent.predict(state)
```

### 4. 風險控制
```python
# 倉位管理
from quantproject.rl_trading.utils.risk_manager import RiskManager

risk_manager = RiskManager(
    max_position_size=0.1,
    max_drawdown=0.15,
    daily_loss_limit=0.05
)

# 檢查交易是否符合風險規則
if risk_manager.validate_trade(trade):
    execute_trade(trade)
```

## 監控與維護

### 1. 系統健康檢查
```python
# 監控系統狀態
python src/integration/health_monitor.py --check-all
```

### 2. 性能優化
```python
# 運行性能分析
python src/optimization/system_profiler.py

# 超參數優化
python src/optimization/hyperparameter_optimizer.py \
    --target lstm \
    --n-trials 100
```

### 3. 日誌管理
```bash
# 查看系統日誌
tail -f logs/system.log

# 查看交易日誌
tail -f logs/trading.log
```

## 常見問題

### Q1: 如何切換交易模式？
```python
# 在 run_system.py 中設置
config = {
    "mode": "paper",  # live/paper/backtest
    "use_real_money": False
}
```

### Q2: 如何添加新的股票？
1. 更新配置文件中的 symbols
2. 下載歷史數據
3. 重新訓練模型（如需要）

### Q3: 如何調整風險參數？
編輯 `config/risk_parameters.yaml` 或在運行時傳入參數

### Q4: 系統崩潰如何恢復？
系統具有斷點續傳功能：
```python
python src/integration/run_system.py --resume
```

## 部署建議

### 1. 雲端部署
```bash
# AWS EC2 部署示例
# 1. 設置 EC2 實例（推薦 t3.large 以上）
# 2. 安裝 Docker
# 3. 使用 Docker Compose 部署

docker-compose up -d
```

### 2. 本地伺服器
- CPU: 4 核心以上
- RAM: 16GB 以上
- GPU: 建議有 NVIDIA GPU（用於深度學習）
- 存儲: 100GB SSD

### 3. 監控設置
- 使用 Prometheus + Grafana 監控系統指標
- 設置 Telegram/Email 告警
- 定期備份模型和交易記錄

## 安全建議

1. **API 金鑰管理**
   - 永遠不要將 API 金鑰提交到版本控制
   - 使用環境變數或加密存儲
   - 定期輪換金鑰

2. **交易限制**
   - 設置每日最大交易次數
   - 設置最大倉位限制
   - 實施緊急停止機制

3. **數據安全**
   - 加密敏感數據
   - 定期備份
   - 實施訪問控制

## 下一步

1. **擴展功能**
   - 添加更多技術指標
   - 整合更多數據源
   - 開發更複雜的策略

2. **優化性能**
   - 使用 Ray 進行分散式訓練
   - 實施模型量化
   - 優化數據管道

3. **社群參與**
   - 提交 Issue 和 PR
   - 分享使用經驗
   - 參與策略討論

---

如有問題，請查看[文檔](./docs)或提交 [Issue](https://github.com/Shelby-Wong0312/QuantProject/issues)