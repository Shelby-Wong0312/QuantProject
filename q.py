# q.py - 完整版本
from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
import time
from datetime import datetime

print("="*50)
print("MT4 CRUDEOIL 連接測試")
print("="*50)

# 初始化連接
print("\n1. 正在連接到 MT4...")
dwx = DWX_ZeroMQ_Connector(_verbose=True)
time.sleep(2)

# 測試心跳
print("\n2. 發送心跳測試...")
dwx._DWX_MTX_SEND_COMMAND_("HEARTBEAT")
time.sleep(1)

# 獲取帳戶資訊
print("\n3. 獲取帳戶資訊...")
account_info = dwx._DWX_MTX_GET_ACCOUNT_INFO_()
if account_info:
    print(f"   帳戶餘額: ${account_info.get('_balance', 'N/A')}")

# 測試 CRUDEOIL
print("\n4. 訂閱 CRUDEOIL...")
dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('CRUDEOIL')
time.sleep(3)

print("\n5. 檢查市場數據...")
if dwx._Market_Data_DB:
    for symbol, data in dwx._Market_Data_DB.items():
        print(f"   {symbol}: {data}")
else:
    print("   未收到數據")
    
    # 嘗試其他可能的名稱
    print("\n6. 嘗試其他原油符號名稱...")
    oil_names = ['CRUDEOIL.', 'CRUDE', 'OIL', 'USOIL', 'WTI']
    
    for name in oil_names:
        print(f"   嘗試 {name}...")
        dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(name)
        time.sleep(2)
        
        if name in dwx._Market_Data_DB:
            print(f"   ✓ 成功！使用 {name}")
            print(f"   價格: {dwx._Market_Data_DB[name]}")
            break
        else:
            dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(name)

# 測試其他符號
print("\n7. 測試基本外匯對...")
test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']

for symbol in test_symbols:
    print(f"\n   測試 {symbol}...")
    dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
    time.sleep(2)
    
    if symbol in dwx._Market_Data_DB:
        print(f"   ✓ {symbol}: {dwx._Market_Data_DB[symbol]}")
    else:
        print(f"   ✗ {symbol}: 無數據")

print("\n" + "="*50)
print("測試完成！")
print("="*50)

# 顯示最終結果
print("\n最終市場數據:")
if dwx._Market_Data_DB:
    for symbol, data in dwx._Market_Data_DB.items():
        print(f"  {symbol}: {data}")
else:
    print("  沒有收到任何市場數據")
    print("\n可能的原因:")
    print("  1. 市場關閉（檢查交易時間）")
    print("  2. 需要在 MT4 開啟對應的圖表")
    print("  3. 符號名稱不正確")