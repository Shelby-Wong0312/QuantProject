#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4連接驗證腳本
快速測試MT4-Python橋接是否正常工作
"""

import sys
import os
import time
import json
from datetime import datetime
from colorama import init, Fore, Style

# 初始化colorama
init()

# 添加項目路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt4_bridge.connector import MT4Connector, create_default_connector
from src.data_pipeline.mt4_data_collector import MT4DataPipeline, MarketData


def print_success(msg):
    print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")


def print_error(msg):
    print(f"{Fore.RED}✗ {msg}{Style.RESET_ALL}")


def print_info(msg):
    print(f"{Fore.CYAN}ℹ {msg}{Style.RESET_ALL}")


def print_warning(msg):
    print(f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}")


def test_basic_connection():
    """測試基本連接"""
    print("\n" + "=" * 60)
    print("步驟1: 測試基本連接")
    print("=" * 60)

    connector = create_default_connector()

    print_info("正在連接到MT4...")
    if connector.connect():
        print_success("成功連接到MT4")

        # 發送心跳測試
        print_info("發送心跳測試...")
        response = connector.send_command("HEARTBEAT")
        if response and response.get("status") == "ok":
            print_success("心跳測試成功")
        else:
            print_error("心跳測試失敗")
            return False

        # 獲取帳戶信息
        print_info("獲取帳戶信息...")
        response = connector.send_command("GET_ACCOUNT_INFO")
        if response and response.get("status") == "ok":
            data = response.get("data", {})
            print_success("帳戶信息獲取成功")
            print(f"  帳號: {data.get('account_number')}")
            print(f"  餘額: ${data.get('balance')}")
            print(f"  槓桿: 1:{data.get('leverage')}")
        else:
            print_error("無法獲取帳戶信息")

        connector.disconnect()
        return True
    else:
        print_error("無法連接到MT4")
        print_warning("請檢查:")
        print("  1. MT4是否已啟動並登入Demo帳戶")
        print("  2. PythonBridge EA是否已載入到圖表")
        print("  3. 自動交易是否已啟用（綠色按鈕）")
        print("  4. ZeroMQ DLL是否正確安裝")
        return False


def test_data_pipeline():
    """測試數據管道"""
    print("\n" + "=" * 60)
    print("步驟2: 測試數據管道")
    print("=" * 60)

    pipeline = MT4DataPipeline()

    print_info("啟動數據管道...")
    if pipeline.start():
        print_success("數據管道已啟動")

        # 訂閱EURUSD
        print_info("訂閱EURUSD...")
        pipeline.subscribe("EURUSD")
        print_success("已訂閱EURUSD")

        # 收集數據5秒
        print_info("收集數據中（5秒）...")

        # 添加回調來顯示收到的數據
        data_count = [0]

        def on_data(data: MarketData):
            data_count[0] += 1
            if data_count[0] == 1:
                print_success(f"收到第一筆數據: {data.symbol} @ {data.bid}/{data.ask}")

        pipeline.add_callback(on_data)

        # 等待數據
        for i in range(5):
            time.sleep(1)
            print(f"  {i+1}/5 秒...", end="\r")

        # 獲取統計
        stats = pipeline.get_stats()
        print(f"\n統計結果:")
        print(f"  總Tick數: {stats['total_ticks']}")
        print(f"  有效Tick數: {stats['valid_ticks']}")
        print(f"  錯誤數: {stats['errors']}")

        if stats["total_ticks"] > 0:
            print_success(f"數據收集成功 (有效率: {stats['validity_rate']:.1%})")
        else:
            print_warning("未收到數據（可能是市場休市）")

        # 停止管道
        pipeline.stop()
        print_success("數據管道已停止")
        return True
    else:
        print_error("無法啟動數據管道")
        return False


def test_market_data():
    """測試市場數據獲取"""
    print("\n" + "=" * 60)
    print("步驟3: 測試市場數據")
    print("=" * 60)

    connector = create_default_connector()

    if connector.connect():
        print_info("獲取市場數據...")

        # 獲取當前價格
        response = connector.send_command("GET_MARKET_DATA", symbol="")
        if response and response.get("status") == "ok":
            data = response.get("data", {})
            print_success("市場數據獲取成功")
            print(f"  交易品種: {data.get('symbol')}")
            print(f"  買價: {data.get('bid')}")
            print(f"  賣價: {data.get('ask')}")
            print(f"  點差: {data.get('spread')} points")
        else:
            print_error("無法獲取市場數據")

        # 獲取持倉
        print_info("獲取當前持倉...")
        response = connector.send_command("GET_POSITIONS")
        if response and response.get("status") == "ok":
            positions = response.get("data", [])
            print_success(f"找到 {len(positions)} 個持倉")
            for pos in positions[:3]:  # 顯示前3個
                print(f"  - {pos.get('symbol')}: {pos.get('lots')} lots @ {pos.get('open_price')}")

        connector.disconnect()
        return True
    else:
        print_error("無法連接到MT4")
        return False


def main():
    print("\n" + "=" * 70)
    print(" MT4-Python 橋接連接驗證工具 ")
    print("=" * 70)

    print_info("開始驗證MT4連接...")
    print_info(f"當前時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 檢查是否為交易時間
    now = datetime.now()
    if now.weekday() >= 5:  # 週六或週日
        print_warning("注意: 現在是週末，外匯市場休市，可能無法收到實時數據")

    # 執行測試
    results = []

    # 測試1: 基本連接
    if test_basic_connection():
        results.append(("基本連接", True))

        # 測試2: 數據管道
        if test_data_pipeline():
            results.append(("數據管道", True))
        else:
            results.append(("數據管道", False))

        # 測試3: 市場數據
        if test_market_data():
            results.append(("市場數據", True))
        else:
            results.append(("市場數據", False))
    else:
        results.append(("基本連接", False))
        print_warning("跳過後續測試")

    # 顯示總結
    print("\n" + "=" * 70)
    print(" 測試總結 ")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        if result:
            print_success(f"{name}: 通過")
        else:
            print_error(f"{name}: 失敗")

    print(f"\n總計: {passed}/{total} 測試通過")

    if passed == total:
        print_success("\n🎉 恭喜！MT4橋接系統運行正常，可以開始使用了！")
        print("\n下一步:")
        print("1. 使用 start_data_collection() 開始收集數據")
        print("2. 使用 get_realtime_data() 獲取實時價格")
        print("3. 使用 get_historical_data() 獲取歷史數據")
    else:
        print_error("\n部分測試失敗，請檢查配置")
        print("\n故障排除:")
        print("1. 確認MT4已啟動並登入")
        print("2. 確認PythonBridge EA已載入並顯示笑臉")
        print("3. 確認自動交易按鈕為綠色")
        print("4. 查看MT4的專家標籤是否有錯誤訊息")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print_error(f"\n發生錯誤: {e}")
        import traceback

        traceback.print_exc()
