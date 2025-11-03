# MT4 橋接設置指南

## 快速開始

### 步驟 1：安裝 Python 依賴

```bash
pip install zmq
pip install pyzmq
```

### 步驟 2：MT4 設置

1. **下載 Capital.com MT4**
   - 訪問 Capital.com 網站下載 MT4
   - 使用您的 Demo 帳戶登入

2. **安裝 ZeroMQ 庫到 MT4**
   - 下載 [mql-zmq](https://github.com/dingmaotu/mql-zmq)
   - 將 `Include/Zmq` 複製到 `MT4/MQL4/Include/`
   - 將 `Libraries/*.dll` 複製到 `MT4/MQL4/Libraries/`

3. **安裝 Expert Advisor**
   - 將 `MT4_EA.mq4` 複製到 `MT4/MQL4/Experts/`
   - 在 MetaEditor 中編譯
   - 在圖表上啟用 EA

### 步驟 3：測試連接

```python
from mt4_bridge.zeromq.python_side import MT4Bridge

# 創建橋接
bridge = MT4Bridge()

# 測試連接
account = bridge.get_account_info()
print(f"帳戶餘額: {account.get('balance')}")
```

## 選擇合適的橋接方案

### ZeroMQ（推薦用於生產環境）
**優點：**
- 延遲最低（< 1ms）
- 支援大量並發
- 穩定可靠

**缺點：**
- 需要安裝額外 DLL
- 設置較複雜

### 檔案通訊（推薦用於開發測試）
**優點：**
- 設置簡單
- 不需額外依賴
- 易於調試

**缺點：**
- 延遲較高（10-100ms）
- 需處理檔案鎖定

## 常見問題

### Q: DLL 載入失敗？
A: 確保 Visual C++ Redistributable 已安裝

### Q: 無法連接到 MT4？
A: 檢查防火牆設置，允許本地端口 5555, 5556

### Q: 延遲太高？
A: 使用 ZeroMQ 方案，避免檔案通訊

## 整合現有策略

```python
# 您現有的策略
class YourStrategy:
    def on_signal(self, signal):
        # 原本：使用 REST API
        # self.capital_api.place_order(...)
        
        # 現在：使用 MT4 橋接
        self.mt4_bridge.place_order(
            symbol=signal.symbol,
            order_type=signal.direction,
            volume=signal.size
        )
```

## 性能優化建議

1. **批量處理**：累積多個命令一起發送
2. **異步操作**：使用異步方法避免阻塞
3. **連接池**：維護多個連接提高吞吐量
4. **本地快取**：快取常用數據減少請求

## 安全考慮

1. **僅在本地使用**：不要暴露端口到網路
2. **加密通訊**：如需網路傳輸，使用 SSL
3. **權限控制**：限制 EA 的交易權限
4. **錯誤處理**：完善的異常處理機制