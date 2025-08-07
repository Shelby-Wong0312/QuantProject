# MT4-Python 橋接方案

## 方案比較

### 1. ZeroMQ 方案（推薦）
- **優點**：低延遲、雙向通訊、穩定可靠
- **缺點**：需要安裝額外 DLL

### 2. 檔案通訊方案
- **優點**：簡單易實施、不需額外依賴
- **缺點**：延遲較高、需處理檔案鎖定

### 3. MetaAPI 雲端方案
- **優點**：不需本地 MT4、REST API 接口
- **缺點**：需付費、依賴網路

## 實作架構

```
Python 策略
    ↓
橋接層（ZeroMQ/File/API）
    ↓
MT4 Expert Advisor
    ↓
Capital.com 伺服器
```

## 檔案結構

```
mt4_bridge/
├── zeromq/          # ZeroMQ 實作
├── file_bridge/     # 檔案通訊實作
├── metaapi/         # MetaAPI 實作
└── examples/        # 使用範例
```