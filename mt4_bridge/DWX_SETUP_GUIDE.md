# DWX ZeroMQ MT4 設置指南

## 📦 系統架構

DWX系統使用三個端口進行通訊：
- **32768**: PUSH (Python → MT4) 發送命令
- **32769**: PULL (MT4 → Python) 接收回應
- **32770**: SUB (MT4 → Python) 市場數據流

## 🛠️ MT4端設置步驟

### 步驟1：獲取DWX Server EA

如果您沒有 `DWX_ZeroMQ_Server_v2.0.1_RC8.mq4`，可以從以下來源獲取：
- [Darwinex官方GitHub](https://github.com/darwinex/dwx-zeromq-connector)
- 或使用我們提供的兼容版本

### 步驟2：安裝EA到MT4

1. **複製EA文件**
   ```
   將 DWX_ZeroMQ_Server_v2.0.1_RC8.mq4 複製到:
   C:\Program Files (x86)\[您的MT4]\MQL4\Experts\
   ```

2. **複製Include文件**（如果有）
   ```
   將任何 .mqh 文件複製到:
   C:\Program Files (x86)\[您的MT4]\MQL4\Include\
   ```

### 步驟3：編譯EA

1. 在MT4中按 `F4` 開啟MetaEditor
2. 打開 `DWX_ZeroMQ_Server_v2.0.1_RC8.mq4`
3. 按 `F7` 編譯
4. 確保顯示 "0 errors, 0 warnings"

### 步驟4：配置並載入EA

1. **打開圖表**
   - 建議使用 EUR/USD M1 圖表

2. **拖拽EA到圖表**
   - 從導航器拖拽 `DWX_ZeroMQ_Server` 到圖表

3. **配置參數**
   ```
   [General]
   ☑ Allow automated trading
   ☑ Allow DLL imports
   
   [Inputs]
   MILLISECOND_TIMER = 1
   PUSH_PORT = 32768
   PULL_PORT = 32769
   PUB_PORT = 32770
   MaximumOrders = 1
   MaximumLotSize = 0.01
   DMA_MODE = true
   ```

4. **點擊OK**

### 步驟5：啟用自動交易

1. 點擊工具欄的「AutoTrading」按鈕（應為綠色）
2. 確認圖表右上角顯示笑臉 😊

### 步驟6：驗證EA運行

在MT4的「Experts」標籤應看到：
```
DWX_ZeroMQ_Server: EA Initialized Successfully
DWX_ZeroMQ_Server: Binding PUSH Socket on Port 32768
DWX_ZeroMQ_Server: Binding PULL Socket on Port 32769
DWX_ZeroMQ_Server: Binding PUB Socket on Port 32770
```

## 🐍 Python端測試

### 快速測試連接

```python
from quantproject.data_pipeline.dwx_data_collector import DWXDataCollector

# 創建收集器
collector = DWXDataCollector()

# 連接到MT4
if collector.connect():
    print("✓ 連接成功")
    
    # 獲取帳戶信息
    info = collector.get_account_info()
    print(f"帳戶餘額: ${info.get('_account_balance', 0)}")
    
    # 訂閱EURUSD
    collector.subscribe('EURUSD')
    
    # 開始收集數據
    collector.start_collection()
    
    # 獲取最新價格
    price = collector.get_latest_price('EURUSD')
    print(f"EUR/USD: {price['bid']}/{price['ask']}")
else:
    print("✗ 連接失敗")
```

### 完整測試

```bash
python src/data_pipeline/dwx_data_collector.py
```

## 🔧 故障排除

### 問題1：無法連接
**症狀**：Python顯示連接超時
**解決方案**：
- 確認EA顯示笑臉
- 檢查端口號是否正確
- 關閉防火牆或添加例外

### 問題2：沒有數據
**症狀**：連接成功但沒有價格數據
**解決方案**：
- 確認市場開市（週一至週五）
- 檢查訂閱的品種名稱是否正確
- 在MT4查看是否有該品種

### 問題3：DLL錯誤
**症狀**：EA顯示DLL載入失敗
**解決方案**：
- 確認ZeroMQ DLL已安裝
- 在EA設置中允許DLL導入
- 重啟MT4

### 問題4：端口被佔用
**症狀**：EA無法綁定端口
**解決方案**：
```cmd
# 檢查端口佔用
netstat -an | findstr 32768
netstat -an | findstr 32769
netstat -an | findstr 32770
```
如果被佔用，關閉佔用的程序或更改端口號

## 📊 使用DWX收集數據

### 基本用法

```python
from quantproject.data_pipeline.dwx_data_collector import DWXDataCollector
import time

# 初始化
collector = DWXDataCollector(
    client_id='MyTradingBot',
    verbose=True
)

# 連接
collector.connect()

# 訂閱多個品種
collector.subscribe(['EURUSD', 'GBPUSD', 'USDJPY'])

# 開始收集
collector.start_collection()

# 添加數據處理回調
def process_tick(symbol, data):
    print(f"{symbol}: {data['bid']}/{data['ask']}")
    # 這裡可以添加策略邏輯

collector.add_tick_callback(process_tick)

# 運行
try:
    while True:
        # 每10秒顯示統計
        time.sleep(10)
        stats = collector.get_stats()
        print(f"收到 {stats['tick_count']} ticks")
except KeyboardInterrupt:
    collector.stop_collection()
    collector.disconnect()
```

### 交易執行

```python
# 下單
collector.place_order(
    symbol='EURUSD',
    order_type='BUY',
    lots=0.01,
    sl=0,  # 停損價格
    tp=0,  # 止盈價格
    comment='Test trade'
)

# 獲取開倉
trades = collector.get_open_trades()

# 平倉
if trades:
    collector.close_position(trades[0]['ticket'])
```

## ✅ 檢查清單

- [ ] MT4已安裝並登入Demo帳戶
- [ ] ZeroMQ DLL已安裝
- [ ] DWX Server EA已編譯無錯誤
- [ ] EA已載入到圖表並顯示笑臉
- [ ] 自動交易按鈕是綠色
- [ ] Expert標籤顯示EA初始化成功
- [ ] 端口32768, 32769, 32770未被佔用
- [ ] Python可以連接並接收數據

## 📚 資源

- [DWX官方文檔](https://github.com/darwinex/dwx-zeromq-connector)
- [ZeroMQ文檔](https://zeromq.org/languages/python/)
- [MT4 MQL4參考](https://docs.mql4.com/)

## 🆘 需要幫助？

運行診斷腳本：
```bash
python test_dwx_connector.py
```

查看詳細日誌：
```python
collector = DWXDataCollector(verbose=True)
```