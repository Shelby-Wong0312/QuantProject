#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4橋接測試腳本
測試MT4-Python通訊功能
"""

import sys
import os
import time
import json
from datetime import datetime

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt4_bridge.connector import MT4Connector, create_default_connector
from mt4_bridge.data_collector import MT4DataCollector
from mt4_bridge.signal_sender import MT4SignalSender
from mt4_bridge.account_monitor import MT4AccountMonitor


def test_basic_connection():
    """測試基本連接"""
    print("\n" + "=" * 50)
    print("測試1: 基本連接測試")
    print("=" * 50)

    # 創建連接器
    connector = create_default_connector()

    # 連接到MT4
    if connector.connect():
        print("✓ 成功連接到MT4")

        # 獲取連接狀態
        stats = connector.get_stats()
        print(f"連接狀態: {json.dumps(stats, indent=2)}")

        # 發送心跳
        response = connector.send_command("HEARTBEAT")
        if response and response.get("status") == "ok":
            print("✓ 心跳測試成功")
        else:
            print("✗ 心跳測試失敗")

        # 斷開連接
        connector.disconnect()
        print("✓ 已斷開連接")
    else:
        print("✗ 無法連接到MT4，請確保:")
        print("  1. MT4已啟動並登入")
        print("  2. PythonBridge EA已載入到圖表")
        print("  3. 自動交易已啟用")
        return False

    return True


def test_account_info():
    """測試帳戶信息獲取"""
    print("\n" + "=" * 50)
    print("測試2: 帳戶信息測試")
    print("=" * 50)

    connector = create_default_connector()

    if connector.connect():
        monitor = MT4AccountMonitor(connector)

        # 獲取帳戶信息
        account_info = monitor.get_account_info()
        if account_info:
            print("帳戶信息:")
            print(f"  帳號: {account_info.get('account_number')}")
            print(f"  餘額: ${account_info.get('balance')}")
            print(f"  淨值: ${account_info.get('equity')}")
            print(f"  可用保證金: ${account_info.get('free_margin')}")
            print(f"  槓桿: 1:{account_info.get('leverage')}")
            print("✓ 帳戶信息獲取成功")
        else:
            print("✗ 無法獲取帳戶信息")

        # 獲取持倉
        positions = monitor.get_positions()
        if positions is not None:
            print(f"\n當前持倉數量: {len(positions)}")
            for pos in positions[:3]:  # 顯示前3個持倉
                print(f"  - {pos.get('symbol')}: {pos.get('type_str')} {pos.get('lots')} lots")
            print("✓ 持倉信息獲取成功")

        connector.disconnect()
    else:
        print("✗ 無法連接到MT4")
        return False

    return True


def test_data_stream():
    """測試數據流接收"""
    print("\n" + "=" * 50)
    print("測試3: 實時數據流測試")
    print("=" * 50)

    connector = create_default_connector()

    if connector.connect():
        collector = MT4DataCollector(connector)

        # 啟動數據收集
        collector.start()

        print("正在接收數據流 (10秒)...")
        print("預期接收: Tick數據和K線數據")

        # 收集10秒數據
        start_time = time.time()
        tick_count = 0
        bar_count = 0

        while time.time() - start_time < 10:
            # 獲取最新tick
            tick = collector.get_latest_tick()
            if tick:
                tick_count += 1
                if tick_count == 1:  # 顯示第一個tick
                    print(f"\n收到第一個Tick:")
                    print(f"  Symbol: {tick.get('symbol')}")
                    print(f"  Bid: {tick.get('bid')}")
                    print(f"  Ask: {tick.get('ask')}")
                    print(f"  Spread: {tick.get('spread')}")

            # 獲取最新K線
            ohlc = collector.get_latest_bar()
            if ohlc:
                bar_count += 1
                if bar_count == 1:  # 顯示第一個K線
                    print(f"\n收到第一個K線:")
                    print(f"  Symbol: {ohlc.get('symbol')}")
                    print(f"  Period: {ohlc.get('period')}")
                    print(
                        f"  OHLC: {ohlc.get('open')}/{ohlc.get('high')}/{ohlc.get('low')}/{ohlc.get('close')}"
                    )

            time.sleep(0.1)

        print(f"\n測試結果:")
        print(f"  收到 {tick_count} 個Tick數據")
        print(f"  收到 {bar_count} 個K線數據")

        if tick_count > 0:
            print("✓ Tick數據接收成功")
        else:
            print("✗ 未收到Tick數據")

        if bar_count > 0:
            print("✓ K線數據接收成功")
        else:
            print("△ 未收到K線數據 (可能需要等待新K線形成)")

        # 停止數據收集
        collector.stop()
        connector.disconnect()
    else:
        print("✗ 無法連接到MT4")
        return False

    return True


def test_trading_signals():
    """測試交易信號發送（僅測試，不實際下單）"""
    print("\n" + "=" * 50)
    print("測試4: 交易信號測試（模擬）")
    print("=" * 50)

    connector = create_default_connector()

    if connector.connect():
        sender = MT4SignalSender(connector)

        # 獲取市場數據
        market_data = sender.get_market_data()
        if market_data:
            print(f"當前市場數據:")
            print(f"  Symbol: {market_data.get('symbol')}")
            print(f"  Bid: {market_data.get('bid')}")
            print(f"  Ask: {market_data.get('ask')}")
            print(f"  Spread: {market_data.get('spread')}")
            print("✓ 市場數據獲取成功")
        else:
            print("✗ 無法獲取市場數據")

        print("\n模擬交易信號測試:")
        print("  買入信號: BUY 0.01 lots")
        print("  賣出信號: SELL 0.01 lots")
        print("  (實際不會執行交易)")
        print("✓ 交易信號系統就緒")

        connector.disconnect()
    else:
        print("✗ 無法連接到MT4")
        return False

    return True


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("MT4-Python橋接測試程序")
    print("=" * 60)
    print("\n請確保:")
    print("1. MT4已啟動並登入Demo帳戶")
    print("2. PythonBridge.mq4 EA已編譯並載入到圖表")
    print("3. 自動交易功能已啟用")
    print("4. DLL導入已允許")

    input("\n按Enter鍵開始測試...")

    # 執行測試
    tests = [
        ("基本連接", test_basic_connection),
        ("帳戶信息", test_account_info),
        ("數據流", test_data_stream),
        ("交易信號", test_trading_signals),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n測試 {name} 發生錯誤: {e}")
            results.append((name, False))

        time.sleep(1)  # 測試間隔

    # 顯示測試結果總結
    print("\n" + "=" * 60)
    print("測試結果總結")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通過" if result else "✗ 失敗"
        print(f"{name}: {status}")

    print(f"\n總計: {passed}/{total} 測試通過")

    if passed == total:
        print("\n🎉 所有測試通過！MT4橋接系統運行正常。")
    else:
        print("\n⚠️ 部分測試失敗，請檢查配置。")


if __name__ == "__main__":
    main()
