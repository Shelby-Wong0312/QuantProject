# 🚀 QuantProject - 智能量化交易系統

一個完整的AI驅動量化交易系統，已成功從MT4遷移到Capital.com API，實現7/24全自動化交易。

## 📊 當前系統狀態
- **平台**: Capital.com REST API (已廢棄MT4)
- **帳戶餘額**: $137,766.45 USD
- **活躍持倉**: 1 BTC @ $116,465.30 (+$0.70)
- **可用市場**: 29個 (加密貨幣、外匯、商品、指數)
- **運行模式**: 7/24全自動化

## 🚀 系統特點

### 核心功能
- **Capital.com API整合**: 完整的REST API連接和交易執行
- **實時數據收集**: 29個市場的即時報價和歷史數據
- **自動交易執行**: 市價單、限價單、止損止盈自動管理
- **AI預測模型**: 整合LSTM、FinBERT情緒分析和GNN關聯分析
- **多策略系統**: 動量、均值回歸、趨勢跟隨策略
- **7/24自動化**: 無需人工干預的全自動交易系統
- **風險管理**: 動態止損、倉位控制、Kelly公式

### 技術架構
```
Capital.com API → REST/WebSocket → Python系統
        ↓                ↓              ↓
   數據收集層      AI決策層      執行管理層
        ↓                ↓              ↓
   實時價格      ML/DL模型      自動下單
   歷史數據      策略信號      風險控制
   帳戶信息      投票機制      績效追蹤
```

## 📋 系統需求

- Python 3.8+
- CUDA支援的GPU（用於深度學習模型）
- 穩定的網路連接

## 🔧 安裝步驟

### 1. 克隆專案
```bash
git clone https://github.com/Shelby-Wong0312/QuantProject.git
cd QuantProject
```

### 2. 創建虛擬環境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安裝依賴
```bash
pip install -r requirements.txt
```

### 4. 環境設定
創建 `.env` 檔案並設定API憑證：
```env
# Capital.com API (Demo Account)
CAPITAL_API_KEY="your_demo_api_key"
CAPITAL_IDENTIFIER="your_email@example.com"
CAPITAL_API_PASSWORD="your_api_password"
CAPITAL_DEMO_MODE="True"

# Alpaca API
ALPACA_API_KEY_ID="your_alpaca_key"
ALPACA_SECRET_KEY="your_alpaca_secret"
ALPACA_PAPER_TRADING="True"

# Other Settings
LOG_LEVEL="INFO"
INITIAL_CAPITAL="100000"
```

## 📁 專案結構

```
QuantProject/
├── core/                    # 核心事件驅動系統
│   ├── event_loop.py       # 異步事件循環
│   └── event.py            # 事件類型定義
├── src/
│   ├── data_pipeline/      # 數據處理和實時數據源
│   │   ├── live_feed.py    # Capital.com實時數據
│   │   └── alpha_research/ # Alpha信號研究
│   ├── strategies/         # 交易策略
│   │   └── trading_strategies.py
│   ├── backtesting/        # 回測引擎
│   └── visualization/      # 視覺化儀表板
├── execution/              # 訂單執行模組
│   ├── portfolio.py        # 投資組合管理
│   └── broker.py           # 經紀商接口
├── config.py               # 系統配置
└── main.py                 # 主程式入口
```

## 🚦 快速開始

### 運行實時交易系統
```bash
python main.py
```

### 運行回測
```bash
python -m src.backtesting.backtest_engine
```

### 啟動視覺化儀表板
```bash
streamlit run src/visualization/dashboard.py
```

## 📊 API設定指南

### Capital.com Demo Account
1. 註冊Demo帳戶: https://demo.capital.com/
2. 啟用2FA（雙重認證）
3. 前往 Settings > API integrations
4. 生成API金鑰並設定專用密碼
5. 更新 `.env` 檔案

### 診斷工具
```bash
# 測試API連接
python test_capital_api.py

# 詳細診斷
python diagnose_api.py
```

## 🤖 AI模型

- **LSTM趨勢預測**: 基於歷史價格數據的時序預測
- **FinBERT情緒分析**: 財經新聞情緒評分
- **GNN關聯分析**: 股票間相關性建模
- **強化學習Agent**: DQN/PPO用於交易決策

## 📈 交易策略

系統支援多種策略：
- 技術指標策略（EMA、RSI、ATR）
- AI預測信號策略
- 強化學習自適應策略
- 多因子Alpha策略

## ⚠️ 風險提醒

- 本系統僅供教育和研究用途
- 實盤交易前請充分測試
- 請勿投入無法承受損失的資金
- 過去績效不代表未來表現

## 🤝 貢獻指南

歡迎提交Issue和Pull Request！

## 📄 授權

MIT License

## 📞 聯絡方式

如有問題請提交GitHub Issue或聯繫專案維護者。

---
*最後更新: 2025年8月*