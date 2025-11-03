# DevOps MT4 交易修復指南

## 🔴 緊急修復步驟

### 步驟 1: 檢查 MT4 狀態
1. **確認 MT4 已啟動**
   - 打開 MT4 (Capital.com MT4)
   - 登入 DEMO 帳戶

2. **檢查連接狀態**
   - 查看右下角連接圖標
   - 應顯示網絡信號強度
   - 如顯示"無連接"，重新登入

### 步驟 2: 配置 EA 自動交易
1. **開啟自動交易**
   - 點擊工具欄的 "AutoTrading" 按鈕
   - 按鈕應變為綠色 ✅
   - 如果是紅色 ❌，點擊一次開啟

2. **設置 EA 權限**
   - 工具 → 選項 → Expert Advisors
   - ✅ 勾選 "Allow automated trading"
   - ✅ 勾選 "Allow DLL imports"
   - ✅ 勾選 "Allow external experts imports"
   - 點擊 OK

### 步驟 3: 載入 DWX EA
1. **打開圖表**
   - 文件 → 新圖表 → EURUSD (或任意符號)

2. **附加 EA**
   - 在導航器找到 DWX_ZeroMQ_Server_v2.0.1_RC8
   - 拖放到圖表上
   - 或雙擊 EA 名稱

3. **配置 EA 屬性**
   - 在彈出窗口中：
   - Common 標籤：
     - ✅ Allow live trading
     - ✅ Allow DLL imports
   - Inputs 標籤：
     - 保持默認設置
   - 點擊 OK

4. **驗證 EA 運行**
   - 圖表右上角應顯示 😊 笑臉
   - 如果是 ☹️，檢查自動交易設置

### 步驟 4: 測試連接
```bash
# 運行 DevOps 測試
python devops_test_basic.py

# 如果成功，運行交易測試
python devops_fixed_trade.py
```

## 🔧 常見問題解決

### 問題 1: "Resource timeout"
**解決方案：**
- 增加超時時間
- 使用 EURUSD 代替 BTCUSD
- 重啟 MT4

### 問題 2: EA 顯示 ☹️
**解決方案：**
1. 按 F7 打開 EA 屬性
2. 勾選所有權限選項
3. 點擊 OK
4. 如果還是不行，移除 EA 並重新附加

### 問題 3: 無法下單
**解決方案：**
1. 檢查帳戶餘額
2. 確認市場開放時間
3. 使用更小的手數 (0.01)
4. 檢查符號是否可交易

## 📊 驗證檢查清單

- [ ] MT4 已啟動並登入
- [ ] AutoTrading 按鈕是綠色
- [ ] EA 顯示笑臉 😊
- [ ] 工具→選項 中所有權限已勾選
- [ ] 可以看到價格更新
- [ ] python devops_test_basic.py 無錯誤

## 🚀 修復後測試

1. **基本連接測試**
   ```bash
   python devops_test_basic.py
   ```

2. **帳戶信息測試**
   ```bash
   python check_and_clean_orders.py
   ```

3. **交易功能測試**
   ```bash
   python devops_fixed_trade.py
   ```

## 📈 優化建議

1. **使用穩定符號**
   - EURUSD (最穩定)
   - GOLD (流動性好)
   - 避免 BTCUSD (可能無權限)

2. **調整參數**
   ```python
   # 優化的連接參數
   dwx = DWX_ZeroMQ_Connector(
       _poll_timeout=3000,    # 增加超時
       _sleep_delay=0.005,    # 優化延遲
       _verbose=False         # 減少輸出
   )
   ```

3. **錯誤處理**
   - 使用 try/except 包裹所有交易操作
   - 實現自動重試機制
   - 記錄所有錯誤日誌

## 💡 DevOps 建議

1. **立即行動**：檢查 MT4 是否運行
2. **5分鐘內**：配置 EA 設置
3. **10分鐘內**：運行測試腳本
4. **15分鐘內**：確認交易功能正常

---
*DevOps Team - 2025-08-07*
*Status: Critical - Requires Manual Intervention*