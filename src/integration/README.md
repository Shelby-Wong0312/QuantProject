# 系統整合模組

## 概述

本模組負責整合所有組件，提供統一的交易系統控制介面。包含主控制器、資料管道、健康監控等核心功能。

## 主要組件

### 1. MainController (主控制器)
- 中央協調器，管理所有子系統
- 支援回測、模擬交易、實盤交易三種模式
- 自動化交易決策流程
- 性能追蹤與報告生成

### 2. DataPipeline (資料管道)
- 管理資料流向各組件
- 實時資料處理與分發
- 技術指標計算
- 資料品質監控

### 3. HealthMonitor (健康監控)
- 組件健康狀態檢查
- 系統資源監控
- 異常警報機制
- 健康報告生成

## 快速開始

### 基本使用

```bash
# 創建預設配置
python src/integration/run_system.py --create-config

# 執行回測模式
python src/integration/run_system.py --mode backtest

# 執行模擬交易
python src/integration/run_system.py --mode paper

# 查看幫助
python src/integration/run_system.py --help
```

### 配置文件

系統配置保存在 `config/system_config.json`：

```json
{
    "mode": "backtest",
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "initial_capital": 100000,
    "risk_limit": 0.02,
    "model_paths": {
        "lstm": "./models/lstm_predictor.h5",
        "rl_agent": "./models/ppo_agent"
    }
}
```

## 系統架構

```
MainController
├── DataClient (Capital.com)
├── DataPipeline
│   ├── Market Data Buffer
│   ├── Feature Engineering
│   └── Real-time Distribution
├── Sensory Models
│   ├── LSTM Predictor
│   └── FinBERT Analyzer
├── RL Trading
│   ├── Trading Environment
│   └── PPO Agent
├── Backtesting Engine
└── Health Monitor
```

## 交易流程

1. **資料收集**: DataClient 獲取市場數據
2. **特徵工程**: DataPipeline 計算技術指標
3. **預測分析**: LSTM 預測趨勢，FinBERT 分析情緒
4. **決策制定**: RL Agent 根據狀態決定交易動作
5. **執行交易**: 根據模式執行回測或真實交易
6. **監控記錄**: 健康監控與性能追蹤

## 監控與日誌

### 健康檢查
- CPU、記憶體、硬碟使用率
- 組件連接狀態
- 資料延遲監控
- 模型性能追蹤

### 日誌文件
- `logs/system.log`: 系統運行日誌
- `logs/trading_decisions.jsonl`: 交易決策記錄
- `logs/health/`: 健康檢查報告
- `alerts/`: 異常警報記錄

### 報告輸出
- `reports/backtest_*.json`: 回測報告
- `reports/performance_*.json`: 性能報告

## 整合測試

執行整合測試：

```bash
python -m pytest tests/integration/test_system_integration.py -v
```

測試覆蓋：
- 組件初始化
- 交易信號處理
- 資料管道功能
- 健康監控機制
- 端到端流程

## 性能優化

1. **並行處理**
   - 使用 ThreadPoolExecutor 處理計算密集任務
   - AsyncIO 處理 I/O 操作

2. **資料快取**
   - 市場數據緩衝區
   - 預測結果快取

3. **資源管理**
   - 自動清理過期數據
   - 記憶體使用優化

## 擴展開發

### 添加新組件

1. 在 MainController 中註冊組件
2. 實現健康檢查方法
3. 整合到交易流程

### 自定義策略

1. 修改 `_process_trading_signal` 方法
2. 添加新的特徵工程
3. 調整風險管理參數

## 注意事項

1. **模型路徑**: 確保模型文件存在於指定路徑
2. **API 憑證**: 實盤交易需要配置 API 金鑰
3. **風險控制**: 謹慎設置 risk_limit 參數
4. **資源需求**: 建議至少 8GB RAM，4 核 CPU

## 故障排除

### 常見問題

1. **組件初始化失敗**
   - 檢查依賴套件安裝
   - 確認模型文件存在
   - 查看錯誤日誌

2. **資料延遲過高**
   - 檢查網路連接
   - 調整 buffer_size
   - 優化特徵計算

3. **記憶體使用過高**
   - 減少歷史數據長度
   - 調整並行任務數
   - 清理舊日誌文件