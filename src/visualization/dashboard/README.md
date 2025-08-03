# 量化交易視覺化儀表板

## 概述

這是一個整合 LSTM 預測和 FinBERT 情緒分析的智能交易信號視覺化儀表板。

## 功能特點

### 1. Alpha 生成頁面
- **K線圖與交易信號**：互動式 K 線圖，標記買入/賣出點
- **LSTM 趨勢預測**：展示 1日、5日、20日預測曲線及置信區間
- **FinBERT 情緒分析**：實時情緒分數儀表盤和新聞情緒展示
- **技術指標對比**：RSI、MACD、布林通道等指標與 ML 預測對比

### 2. 實時數據更新
- 支援自動刷新（可調整頻率）
- WebSocket 支援（未來版本）

### 3. 多股票支援
- 快速切換不同股票
- 自定義時間範圍

## 安裝指南

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 運行儀表板：
```bash
python app.py
```

3. 打開瀏覽器訪問：
```
http://localhost:8050
```

## 使用說明

### 基本操作
1. **選擇股票**：從下拉菜單選擇要分析的股票
2. **設定時間範圍**：選擇 1個月、3個月、6個月或1年
3. **調整更新頻率**：選擇實時、5秒、30秒或1分鐘更新
4. **查看信號**：綜合信號面板顯示買入/賣出建議

### 數據整合

儀表板自動從以下模組載入數據：
- `data_processing/`：處理後的市場數據
- `models/ml_models/`：LSTM 預測結果
- `models/sentiment/`：FinBERT 情緒分析
- `backtesting/`：回測交易信號

如果實際數據不可用，會自動使用模擬數據進行演示。

## 技術架構

### 核心技術
- **前端框架**：Plotly Dash
- **UI 組件**：Dash Bootstrap Components
- **圖表庫**：Plotly
- **數據處理**：Pandas, NumPy

### 目錄結構
```
dashboard/
├── app.py                 # 主應用程式
├── pages/
│   └── alpha_generation.py # Alpha生成頁面
├── components/            # 可重用組件
│   ├── candlestick_chart.py
│   ├── lstm_prediction.py
│   ├── sentiment_panel.py
│   └── indicators_comparison.py
├── data_loader.py        # 數據載入介面
├── requirements.txt      # 依賴清單
└── README.md            # 本文檔
```

## 開發指南

### 添加新頁面
1. 在 `pages/` 目錄創建新的頁面模組
2. 在 `app.py` 中註冊新頁面
3. 添加相應的導航標籤

### 自定義組件
1. 在 `components/` 目錄創建新組件
2. 遵循現有組件的結構模式
3. 確保組件可重用

### 數據源整合
修改 `data_loader.py` 以整合新的數據源：
```python
def load_custom_data(self, symbol, **kwargs):
    # 實現自定義數據載入邏輯
    pass
```

## 效能優化

- 使用數據快取減少重複載入
- 限制圖表數據點數量
- 使用 `dcc.Store` 組件存儲共享數據
- 啟用生產模式：`app.run(debug=False)`

## 未來計劃

- [ ] WebSocket 實時數據流
- [ ] 更多技術指標
- [ ] 策略回測結果展示
- [ ] 風險管理儀表板
- [ ] 多策略對比分析
- [ ] 導出 PDF 報告功能

## 故障排除

### 常見問題

1. **ImportError**：確保所有依賴已安裝
2. **數據載入失敗**：檢查數據檔案路徑
3. **圖表顯示異常**：清除瀏覽器快取

### 日誌查看
儀表板使用 Python logging 模組，可通過設定日誌級別查看詳細資訊：
```python
logging.basicConfig(level=logging.DEBUG)
```

## 聯繫資訊

如有問題或建議，請提交 Issue 或 Pull Request。