# 📚 檔案功能對照表

## 🎯 核心檔案 (根目錄)

| 檔案名稱 | 功能 | 使用場景 |
|---------|------|----------|
| **live_trading_system_full.py** | 完整自動交易系統 | 生產環境，監控40支股票 |
| **simple_trading_system.py** | 簡化交易系統 | 測試環境，快速驗證 |
| **monitor_trading.py** | 統一監控面板 | 監控交易狀態 |
| **live_dashboard.py** | 視覺化儀表板 | 圖形化顯示 |
| **main_trading.py** | 主程式入口 | 統一啟動點 |
| **final_system_test.py** | 系統測試 | 驗證所有組件 |

## 🔧 批次檔案

| 檔案名稱 | 功能 | 說明 |
|---------|------|------|
| **START_TRADING.bat** | 啟動交易 | 單獨啟動交易系統 |
| **MONITOR_TRADING.bat** | 啟動監控 | 單獨啟動監控面板 |
| **LAUNCH_ALL.bat** | 完整啟動 | 同時啟動交易+監控 |

## 📂 目錄功能

### src/ - 源代碼
- **core/** - 核心交易引擎
- **strategies/** - 交易策略
- **risk/** - 風險管理
- **ml_models/** - 機器學習
- **rl_trading/** - 強化學習(PPO)
- **connectors/** - API連接器
- **indicators/** - 技術指標
- **backtesting/** - 回測系統
- **data/** - 資料處理

### data/ - 資料存儲
- **live_trades_full.db** - 實時交易記錄
- **stock_data_complete.db** - 歷史股票資料
- **quant_trading.db** - 主資料庫
- **csv/** - CSV格式資料
- **parquet/** - Parquet格式資料
- **minute/** - 分鐘級資料

### reports/ - 報告與模型
- **ml_models/ppo_trader_final.pt** - PPO訓練模型
- **backtest/** - 回測報告
- **.json/.txt** - 系統報告

### config/ - 配置
- **api_config.json** - API設定
- **db_config.json** - 資料庫設定
- **.env** - 環境變數

### tests/ - 測試套件
- **integration/** - 整合測試
- **test_*.py** - 單元測試

### examples/ - 範例
- **demo_complete.py** - 完整範例
- **run_demo.py** - 簡單範例

### logs/ - 日誌
- **live_trading_full.log** - 交易日誌
- **trading_system.log** - 系統日誌

## 🔍 常用測試檔案

| 測試類型 | 檔案位置 | 用途 |
|---------|---------|------|
| API測試 | tests/test_capital_api.py | 測試Capital.com連接 |
| 系統測試 | final_system_test.py | 測試完整系統 |
| 資料測試 | tests/test_data_loader.py | 測試資料載入 |
| 策略測試 | tests/test_strategies.py | 測試交易策略 |

## 💡 快速指令

### 測試單一功能
```python
# 測試API連接
python tests/test_capital_api.py

# 測試風險管理
python tests/test_risk_manager.py

# 測試信號生成
python tests/test_signal_generator.py
```

### 檢查系統狀態
```python
# 快速狀態檢查
python final_system_test.py

# 查看資料庫
sqlite3 data/live_trades_full.db ".tables"
```

### 故障排除
```python
# 檢查API憑證
python -c "import os; print(os.environ.get('CAPITAL_API_KEY'))"

# 檢查PPO模型
python -c "import torch; model=torch.load('reports/ml_models/ppo_trader_final.pt', weights_only=False); print('Model OK')"
```

## 📝 檔案命名規則

- **test_*.py** - 測試檔案
- **run_*.py** - 執行腳本
- **check_*.py** - 檢查工具
- **monitor_*.py** - 監控工具
- **start_*.py** - 啟動腳本
- **.bat** - Windows批次檔
- **.md** - 文檔檔案

---
最後更新: 2024-08-14