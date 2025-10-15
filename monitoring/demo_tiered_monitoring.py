#!/usr/bin/env python3
"""
分層監控系統示例 - 簡化版演示
Tiered Monitoring System Demo - Simplified Version
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加項目根目錄到路徑
import sys

sys.path.append(str(Path(__file__).parent.parent))

from monitoring.tiered_monitor import TieredMonitor, TierLevel

# 配置日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("分層監控系統基本使用示例")
    print("=" * 60)

    # 1. 創建監控系統
    print("\n1. 創建分層監控系統...")
    monitor = TieredMonitor()

    # 2. 查看初始配置
    print("\n2. 查看初始配置:")
    status = monitor.get_monitoring_status()
    for tier, count in status["tier_counts"].items():
        print(f"   {tier}: {count} 支股票")

    # 3. 查看層級詳情
    print("\n3. 各層級詳情:")
    details = monitor.get_tier_details()
    for tier_name, tier_data in details.items():
        config = tier_data["config"]
        print(f"   {tier_name}:")
        print(f"     - 更新間隔: {config['update_interval']} 秒")
        print(f"     - 最大股票數: {config['max_symbols']}")
        print(f"     - 優先級: {config['priority']}")
        if tier_data["top_stocks"]:
            top_5 = [s["symbol"] for s in tier_data["top_stocks"][:5]]
            print(f"     - 示例股票: {', '.join(top_5)}")

    # 4. 啟動監控
    print("\n4. 啟動分層監控...")
    monitor.start_monitoring()

    # 5. 運行一段時間觀察
    print("\n5. 監控運行中（30秒）...")
    for i in range(6):
        time.sleep(5)
        current_status = monitor.get_monitoring_status()
        print(
            f"   第{i*5+5}秒: 掃描 {current_status['performance_stats']['total_scans']} 次, "
            f"發現 {current_status['performance_stats']['total_signals']} 個信號"
        )

    # 6. 停止監控
    print("\n6. 停止監控...")
    monitor.stop_monitoring()

    # 7. 生成報告
    print("\n7. 生成監控報告...")
    report_file = monitor.save_monitoring_report()
    print(f"   報告已保存: {report_file}")

    # 8. 最終統計
    final_status = monitor.get_monitoring_status()
    print(f"\n8. 最終統計:")
    print(f"   總掃描次數: {final_status['performance_stats']['total_scans']}")
    print(f"   總信號數量: {final_status['performance_stats']['total_signals']}")
    print(f"   層級調整: {final_status['performance_stats']['tier_adjustments']}")

    print("\n[SUCCESS] 基本使用示例完成！")


def demo_signal_scanning():
    """信號掃描演示"""
    print("\n" + "=" * 60)
    print("信號掃描功能演示")
    print("=" * 60)

    from monitoring.signal_scanner import SignalScanner

    # 創建信號掃描器
    scanner = SignalScanner()

    # 測試股票
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    print(f"\n掃描 {len(test_symbols)} 支測試股票的信號...")

    all_signals = []
    for symbol in test_symbols:
        print(f"\n掃描 {symbol}:")

        # 全面掃描
        signals = scanner.scan_symbol_comprehensive(symbol)

        if signals:
            for signal in signals:
                print(
                    f"  ✓ {signal.signal_type}: {signal.direction} "
                    f"(強度: {signal.strength:.2f})"
                )

            # 計算組合信號強度
            combined_strength = scanner.calculate_combined_signal_strength(signals)
            print(f"  → 組合信號強度: {combined_strength:.2f}")

            all_signals.extend(signals)
        else:
            print(f"  ✗ 無信號檢測到")

    print(f"\n總共檢測到 {len(all_signals)} 個信號")

    # 按信號類型統計
    signal_types = {}
    for signal in all_signals:
        signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1

    if signal_types:
        print("\n信號類型統計:")
        for signal_type, count in signal_types.items():
            print(f"  {signal_type}: {count}")


def demo_tier_adjustment():
    """層級調整演示"""
    print("\n" + "=" * 60)
    print("層級調整機制演示")
    print("=" * 60)

    monitor = TieredMonitor()

    # 顯示初始分佈
    initial_status = monitor.get_monitoring_status()
    print("\n初始層級分佈:")
    for tier, count in initial_status["tier_counts"].items():
        print(f"  {tier}: {count} 支股票")

    # 啟動監控
    monitor.start_monitoring()

    # 模擬強信號觸發升級
    print("\n模擬強信號觸發層級調整...")

    # 選擇一些B級股票模擬強信號
    b_tier_stocks = list(monitor.tier_stocks[TierLevel.B_TIER])[:3]

    for symbol in b_tier_stocks:
        if symbol in monitor.stock_tiers:
            stock_info = monitor.stock_tiers[symbol]
            # 模擬檢測到強信號
            stock_info.signal_strength = 0.95
            stock_info.signal_count = 5
            stock_info.promotion_score = 0.9
            print(f"  為 {symbol} 模擬強信號 (強度: 0.95)")

    # 等待層級調整
    print("\n等待層級調整發生...")
    time.sleep(60)  # 等待1分鐘

    # 檢查調整結果
    final_status = monitor.get_monitoring_status()
    print("\n調整後層級分佈:")
    for tier, count in final_status["tier_counts"].items():
        change = count - initial_status["tier_counts"][tier]
        change_str = f"({change:+d})" if change != 0 else ""
        print(f"  {tier}: {count} 支股票 {change_str}")

    adjustments = final_status["performance_stats"]["tier_adjustments"]
    print(f"\n總計進行了 {adjustments} 次層級調整")

    monitor.stop_monitoring()


def demo_performance_monitoring():
    """性能監控演示"""
    print("\n" + "=" * 60)
    print("性能監控演示")
    print("=" * 60)

    monitor = TieredMonitor()

    print("\n啟動性能監控...")
    monitor.start_monitoring()

    # 監控2分鐘性能
    monitoring_duration = 120  # 2分鐘
    interval = 15  # 每15秒報告一次

    print(f"\n監控 {monitoring_duration} 秒，每 {interval} 秒報告性能...")

    for i in range(0, monitoring_duration, interval):
        time.sleep(interval)

        status = monitor.get_monitoring_status()
        stats = status["performance_stats"]

        # 計算速率
        uptime = status["uptime_seconds"]
        scan_rate = stats["total_scans"] / uptime if uptime > 0 else 0
        signal_rate = stats["total_signals"] / uptime if uptime > 0 else 0

        print(
            f"  第{i+interval}秒: "
            f"掃描速率 {scan_rate:.1f}/秒, "
            f"信號速率 {signal_rate:.2f}/秒, "
            f"活躍線程 {status['active_threads']}"
        )

    # 最終性能報告
    final_status = monitor.get_monitoring_status()
    final_stats = final_status["performance_stats"]
    uptime = final_status["uptime_seconds"]

    print(f"\n最終性能報告:")
    print(f"  運行時間: {uptime:.1f} 秒")
    print(f"  總掃描次數: {final_stats['total_scans']}")
    print(f"  總信號數量: {final_stats['total_signals']}")
    print(f"  平均掃描速率: {final_stats['total_scans']/uptime:.1f} 次/秒")
    print(f"  平均信號速率: {final_stats['total_signals']/uptime:.2f} 個/秒")
    print(f"  層級調整次數: {final_stats['tier_adjustments']}")

    monitor.stop_monitoring()


def main():
    """主演示函數"""
    print("分層監控系統完整演示")
    print("Tiered Monitoring System Complete Demo")
    print("=" * 60)

    try:
        # 1. 基本使用
        demo_basic_usage()

        # 2. 信號掃描
        demo_signal_scanning()

        # 3. 層級調整
        demo_tier_adjustment()

        # 4. 性能監控
        demo_performance_monitoring()

        print("\n" + "=" * 60)
        print("[SUCCESS] 所有演示完成！")
        print("=" * 60)
        print("\n系統特點總結:")
        print("✓ 三層監控架構 (S/A/B級)")
        print("✓ 智能信號掃描")
        print("✓ 自動層級調整")
        print("✓ 高性能批量處理")
        print("✓ 實時性能監控")
        print("✓ 支援4000+股票")

    except Exception as e:
        logger.error(f"演示執行錯誤: {e}")
        print(f"\n[ERROR] 演示執行失敗: {e}")


if __name__ == "__main__":
    main()
