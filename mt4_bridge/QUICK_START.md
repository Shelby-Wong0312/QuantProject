# MT4 快速開始指南

## ✅ 您已完成的步驟
- ZeroMQ DLL 安裝完成

## 📋 接下來的步驟

### 1. 在MT4中安裝EA

1. **複製EA文件到MT4**
   - 找到您的MT4安裝目錄（通常在 `C:\Program Files (x86)\Capital.com MT4\`）
   - 將 `mt4_bridge\mql4\PythonBridge.mq4` 複製到 `MQL4\Experts\` 資料夾

2. **編譯EA**
   - 在MT4中按 F4 開啟MetaEditor
   - 開啟 `PythonBridge.mq4`
   - 按 F7 編譯（應該顯示 "0 errors, 0 warnings"）

3. **載入EA到圖表**
   - 返回MT4主界面
   - 開啟任意圖表（建議EUR/USD）
   - 從導航器拖拽 `PythonBridge` 到圖表
   - 在設定中確保：
     - ✅ 允許自動交易
     - ✅ 允許DLL導入
   - 點擊確定

4. **啟用自動交易**
   - 點擊工具列的「自動交易」按鈕（應變成綠色）
   - 圖表右上角應顯示笑臉 😊

### 2. 驗證連接

運行驗證腳本：

```bash
cd C:\Users\niuji\Documents\QuantProject
python mt4_bridge\verify_connection.py
```

如果一切正常，您應該看到：
- ✅ 基本連接：通過
- ✅ 數據管道：通過
- ✅ 市場數據：通過

### 3. 開始使用

#### 方法1：命令行快速測試

```python
# 在Python中
from src.data_pipeline import start_data_collection, get_realtime_data

# 啟動數據收集
pipeline = start_data_collection(['EURUSD', 'GBPUSD'])

# 獲取實時數據
data = get_realtime_data('EURUSD')
print(f"EUR/USD: {data.bid}/{data.ask}")
```

#### 方法2：運行完整測試

```bash
python mt4_bridge\test_mt4_bridge.py
```

### 4. 檢查數據流

查看實時數據：

```python
from src.data_pipeline import MT4DataPipeline

# 創建管道
pipeline = MT4DataPipeline()
pipeline.start()
pipeline.subscribe('EURUSD')

# 添加回調顯示數據
def show_data(data):
    print(f"{data.symbol}: Bid={data.bid}, Ask={data.ask}, Spread={data.spread}")
    if data.indicators:
        print(f"  RSI={data.indicators.get('rsi14', 'N/A')}")

pipeline.add_callback(show_data)

# 運行10秒
import time
time.sleep(10)

# 查看統計
stats = pipeline.get_stats()
print(f"收到 {stats['total_ticks']} 個tick")
```

## 🔧 故障排除

### 問題：無法連接
- 確認MT4已登入Demo帳戶
- 確認EA顯示笑臉 😊
- 確認自動交易按鈕是綠色

### 問題：沒有數據
- 檢查是否為交易時間（週一至週五）
- 查看MT4專家標籤的錯誤訊息
- 確認訂閱的交易品種正確

### 問題：DLL錯誤
- 確認 `libzmq.dll` 在 `MQL4\Libraries\` 資料夾
- 使用32位版本的DLL（MT4是32位）
- 在MT4設定中允許DLL導入

## 📊 監控面板

查看系統狀態：

```python
from src.data_pipeline import get_pipeline

pipeline = get_pipeline()
stats = pipeline.get_stats()

print("系統狀態:")
print(f"  運行中: {stats['running']}")
print(f"  已連接: {stats['connected']}")
print(f"  訂閱品種: {stats['subscribed_symbols']}")
print(f"  總Tick數: {stats['total_ticks']}")
print(f"  每秒Tick: {stats['ticks_per_second']:.2f}")
print(f"  數據有效率: {stats['validity_rate']:.1%}")
```

## 🚀 下一步

1. **測試交易執行**（當市場開市時）
2. **配置更多交易品種**
3. **調整數據質量參數**
4. **整合到AI策略系統**

## 📞 需要幫助？

如果遇到問題：
1. 運行 `verify_connection.py` 查看詳細診斷
2. 查看 `MT4_SETUP_GUIDE.md` 的故障排除章節
3. 檢查 `logs/` 資料夾的日誌文件