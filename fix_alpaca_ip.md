# 🔧 修復Alpaca IP Allowlist問題

## 🎯 問題診斷
你的Alpaca API返回 **403 Forbidden** 錯誤，很可能是因為 **IP Allowlist (CIDR)** 設置問題。

## 📍 什麼是IP Allowlist？
IP Allowlist是一個安全功能，只允許特定IP地址訪問你的API。如果設置為 **Disabled**，可能會阻擋所有請求。

## ✅ 解決方案

### 方法1：完全開放（最簡單）
1. 登入 https://app.alpaca.markets/
2. 切換到 **Paper Trading** 模式
3. 進入 **API Keys** 頁面
4. 找到 **IP Allowlist** 設置
5. 選擇 **Enable** 並設置為：
   ```
   0.0.0.0/0
   ```
   這會允許所有IP訪問（Paper Trading安全風險較低）

### 方法2：添加你的IP（更安全）
1. 先查詢你的公網IP：
   - 訪問 https://whatismyipaddress.com/
   - 或在命令行運行下面的腳本

2. 在Alpaca設置中：
   - **Enable** IP Allowlist
   - 添加你的IP，格式：`你的IP/32`
   - 例如：`123.456.789.0/32`

### 方法3：添加IP範圍
如果你的IP會變動（動態IP），可以添加範圍：
```
例如：123.456.0.0/16
這會允許 123.456.*.* 的所有IP
```

## 🔍 檢查你的當前IP

運行這個腳本查看你的公網IP：