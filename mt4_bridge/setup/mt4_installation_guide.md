# Capital.com MT4 安裝指南

## 概述
本指南將協助您下載、安裝和設置 Capital.com MetaTrader 4 (MT4) 交易平台。MT4 是一個專業的交易軟體，支援自動化交易、技術分析和多種交易工具。

## 系統需求

### Windows
- 作業系統：Windows 7/8/10/11 (32-bit 或 64-bit)
- 記憶體：至少 512 MB RAM (建議 1 GB 以上)
- 硬碟空間：至少 50 MB 可用空間
- 網路連線：穩定的網際網路連線
- 處理器：1 GHz 或更快的處理器

### Mac
- 作業系統：macOS 10.12 或更新版本
- 記憶體：至少 512 MB RAM (建議 1 GB 以上)
- 硬碟空間：至少 50 MB 可用空間
- 網路連線：穩定的網際網路連線

### 其他平台
- Android：Android 4.0 或更新版本
- iOS：iOS 9.0 或更新版本
- 網頁版：支援所有現代瀏覽器

## 下載連結

### 官方下載頁面
- **主要下載頁面**：https://capital.com/en-gb/trading-platforms/mt4
- **Capital.com MT4 專頁**：https://capital.com/metatrader4-lp

### 平台特定下載
- **Windows**：透過 Capital.com 官網下載 Windows 執行檔
- **Mac**：透過 Capital.com 官網下載 Mac 版本
- **Android**：Google Play Store 搜尋 "MetaTrader 4"
- **iOS**：App Store 搜尋 "MetaTrader 4"
- **網頁版**：直接在瀏覽器中使用，無需下載

## 安裝步驟

### 第一步：建立 Capital.com 帳戶
1. 前往 Capital.com 官方網站
2. 註冊新帳戶並完成身份驗證流程
3. 確保帳戶已啟用並可以進行交易

### 第二步：創建 MT4 帳戶

#### 透過手機應用程式：
1. 開啟 Capital.com 應用程式
2. 點擊 "Account" (帳戶)
3. 點擊 "My account" (我的帳戶)
4. 點擊 "Add Live account" (新增真實帳戶)
5. 選擇 "Capital.com MT4"
6. 點擊 "Continue" (繼續)

#### 透過網頁平台：
1. 登入 Capital.com 網頁平台
2. 點擊 "Settings" (設定)
3. 在 "My accounts" (我的帳戶) 中選擇 "Live accounts" 或 "Demo accounts" 標籤
4. 點擊 "Add Live account" (或 "Add Demo account")
5. 在 "Type" 下拉選單中選擇相關的 MT4 帳戶類型
6. 從下拉選單中選擇您的貨幣
7. 點擊 "Create" (建立)

### 第三步：下載 MT4 軟體

#### Windows 安裝：
1. 從 Capital.com 下載 Windows MT4 安裝程式
2. 執行下載的 .exe 檔案
3. 按照安裝精靈的指示完成安裝
4. 安裝完成後，MT4 將自動啟動

#### Mac 安裝：
1. 從 Capital.com 下載 Mac MT4 安裝程式
2. 開啟下載的 .dmg 檔案
3. 將 MetaTrader 4 拖曳到 Applications 資料夾
4. 從 Applications 資料夾啟動 MetaTrader 4

### 第四步：登入 MT4
1. 啟動 MT4 應用程式
2. 在登入視窗中輸入以下資訊：
   - **伺服器**：
     - 真實帳戶：`Capital.com-Real`
     - 模擬帳戶：`Capital.com-Demo`
   - **登入ID**：建立 MT4 帳戶時提供的登入憑證
   - **密碼**：與您的 Capital.com 平台登入密碼相同
3. 點擊 "Login" (登入)

## Demo 帳戶設置流程

### 建立 Demo 帳戶
1. 按照上述步驟建立帳戶，但選擇 "Demo account" (模擬帳戶)
2. Demo 帳戶將提供虛擬資金供您練習交易
3. 使用伺服器：`Capital.com-Demo`

### Demo 帳戶優勢
- **無風險練習**：使用虛擬資金，沒有實際損失風險
- **完整功能**：擁有真實帳戶的所有交易功能
- **即時市場數據**：存取真實的市場價格和圖表
- **策略測試**：測試交易策略和自動化交易系統

## MT4 平台功能

### 核心功能
- **30+ 技術指標**：移動平均線、RSI、MACD 等
- **多種圖表類型**：線圖、柱狀圖、K線圖
- **自訂圖表設置**：個人化您的交易界面
- **專家顧問 (Expert Advisors)**：自動化交易策略
- **腳本支援**：自訂交易腳本和指標

### 交易工具
- **即時報價**：即時市場價格更新
- **訂單管理**：多種訂單類型和管理工具
- **風險管理**：停損和止盈設置
- **歷史數據**：詳細的交易歷史和分析

## 目錄結構設置

安裝完成後，MT4 將建立以下目錄結構：
```
C:\Program Files\MetaTrader 4\
├── MQL4\
│   ├── Experts\          # Expert Advisors (EA)
│   ├── Indicators\       # 自訂指標
│   ├── Scripts\          # 交易腳本
│   ├── Include\          # 標頭檔案
│   └── Libraries\        # 函式庫
├── Profiles\             # 圖表設定檔
├── Templates\            # 圖表模板
└── Tester\              # 策略測試器資料
```

## 常見問題解決

### 連線問題
1. 檢查網路連線
2. 確認伺服器設置正確
3. 聯絡 Capital.com 客服確認帳戶狀態

### 登入問題
1. 確認登入憑證正確
2. 檢查伺服器名稱是否正確
3. 重設密碼（如需要）

### 平台問題
1. 重新啟動 MT4
2. 重新安裝軟體
3. 檢查防火牆和防毒軟體設置

## 重要注意事項

⚠️ **風險警告**：根據 Capital.com 的統計，64% 的零售投資者帳戶在與該提供商交易價差合約和CFD時會虧損。請確保您了解相關風險。

## 支援和協助

- **Capital.com 客服**：https://help.capital.com/
- **MT4 專區**：https://help.capital.com/hc/en-us/sections/4404600202386-MetaTrader-4-MT4
- **技術支援**：透過 Capital.com 平台內的即時聊天功能

## 下一步

安裝完成後，建議執行：
1. `check_mt4_env.py` - 檢查 MT4 環境設置
2. `prepare_mt4_folders.py` - 準備必要的目錄結構
3. 測試 Demo 帳戶連線
4. 熟悉平台基本操作

---
*本指南最後更新：2025-08-06*