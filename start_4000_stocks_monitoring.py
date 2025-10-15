"""
4000+ 股票大規模監控交易系統
專門設計用於監控和交易4000檔以上股票
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


async def main():
    """主程序：啟動4000+股票監控系統"""

    print("\n" + "=" * 80)
    print("[STARTING] 4000+ Stock Large-Scale Monitoring System")
    print("=" * 80)

    try:
        # 1. 導入必要的模組
        print("\n[1/6] 載入系統模組...")
        from monitoring.tiered_monitor import TieredMonitor
        from data_pipeline.free_data_client import FreeDataClient
        from src.indicators.indicator_calculator import IndicatorCalculator
        from monitoring.signal_scanner import SignalScanner

        print("   ✓ 核心模組載入成功")

        # 2. 初始化數據客戶端
        print("\n[2/6] 初始化數據管道...")
        data_client = FreeDataClient()
        print("   ✓ 數據管道就緒 (Yahoo Finance + Alpha Vantage)")

        # 3. 獲取股票列表
        print("\n[3/6] 載入股票列表...")

        # 獲取S&P 500股票列表作為起點
        sp500_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "BRK-B",
            "NVDA",
            "JPM",
            "JNJ",
            "V",
            "PG",
            "UNH",
            "HD",
            "MA",
            "DIS",
            "BAC",
            "ADBE",
            "CRM",
            "NFLX",
            "PFE",
            "CMCSA",
            "KO",
            "PEP",
            "TMO",
            "CSCO",
            "ABT",
            "NKE",
            "CVX",
            "WMT",
            "ACN",
            "MRK",
            "COST",
            "WFC",
            "VZ",
            "DHR",
            "TXN",
            "INTC",
            "T",
            "MS",
            "UNP",
            "BMY",
            "MDT",
            "LIN",
            "QCOM",
            "LOW",
            "HON",
            "PM",
            "AMGN",
            "IBM",
        ]  # 前50檔作為示範

        # 如果要監控更多股票，可以從文件載入
        all_symbols_file = Path("data/all_symbols.txt")
        if all_symbols_file.exists():
            with open(all_symbols_file, "r") as f:
                all_symbols = [line.strip() for line in f.readlines()]
                print(f"   ✓ 從文件載入 {len(all_symbols)} 檔股票")
        else:
            # 使用示範股票列表
            all_symbols = sp500_symbols
            print(f"   ✓ 使用示範列表 {len(all_symbols)} 檔股票")

        # 4. 初始化分層監控系統
        print("\n[4/6] 初始化分層監控系統...")

        # 創建或更新配置文件
        config_path = Path("monitoring/config.yaml")
        if not config_path.exists():
            # 如果配置文件不存在，創建默認配置
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "monitoring": {
                    "tiers": {
                        "S": {
                            "max_symbols": 40,
                            "update_interval": 1,  # 1秒更新
                            "indicators": ["RSI", "MACD", "BB"],
                            "timeframes": ["1m", "5m", "1h"],
                        },
                        "A": {
                            "max_symbols": 100,
                            "update_interval": 60,  # 1分鐘更新
                            "indicators": ["RSI", "MACD"],
                            "timeframes": ["5m", "1h"],
                        },
                        "B": {
                            "max_symbols": 4000,
                            "update_interval": 300,  # 5分鐘更新
                            "indicators": ["RSI"],
                            "timeframes": ["1d"],
                        },
                    }
                }
            }
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(config, f)
            print("   ✓ 創建了默認配置文件")

        # 使用配置文件路徑初始化監控器
        monitor = TieredMonitor(str(config_path))

        # 初始化股票到B層（全市場掃描）
        from monitoring.tiered_monitor import TierLevel

        for i, symbol in enumerate(all_symbols):
            if i < 40:
                # 前40檔放入S層
                monitor._add_stock_to_tier(symbol, TierLevel.S_TIER)
            elif i < 140:
                # 接下來100檔放入A層
                monitor._add_stock_to_tier(symbol, TierLevel.A_TIER)
            else:
                # 其餘放入B層
                monitor._add_stock_to_tier(symbol, TierLevel.B_TIER)

        # 獲取監控狀態
        status = monitor.get_monitoring_status()
        # 兼容不同版本的鍵名
        dist = (
            status.get("tier_distribution")
            or status.get("stock_allocation")
            or status.get("tier_counts")
            or {}
        )
        # 兼容不同鍵值寫法
        s = dist.get("S_tier", dist.get("s_tier", 0))
        a = dist.get("A_tier", dist.get("a_tier", 0))
        b = dist.get("B_tier", dist.get("b_tier", 0))
        print(f"   ✓ 分層監控系統就緒")
        print(f"      S層: {s} 檔 (實時監控)")
        print(f"      A層: {a} 檔 (高頻監控)")
        print(f"      B層: {b} 檔 (全市場掃描)")

        # 5. 初始化信號掃描器
        print("\n[5/6] 初始化信號掃描系統...")
        signal_scanner = SignalScanner(str(config_path))
        print("   ✓ 信號掃描器就緒")

        # 6. 啟動監控循環
        print("\n[6/6] 啟動監控系統...")
        print("\n" + "=" * 80)
        print("🎯 系統運行中 - 監控 {} 檔股票".format(len(all_symbols)))
        print("按 Ctrl+C 停止系統")
        print("=" * 80 + "\n")

        # 啟動監控系統
        monitor.start_monitoring()

        # 主循環 - 顯示狀態
        scan_counter = 0
        while True:
            scan_counter += 1

            # 每10秒顯示一次狀態
            if scan_counter % 10 == 0:
                status = monitor.get_monitoring_status()
                # 兼容不同版本的鍵名
                dist = (
                    status.get("tier_distribution")
                    or status.get("stock_allocation")
                    or status.get("tier_counts")
                    or {}
                )
                # 兼容不同鍵值寫法
                s = dist.get("S_tier", dist.get("s_tier", 0))
                a = dist.get("A_tier", dist.get("a_tier", 0))
                b = dist.get("B_tier", dist.get("b_tier", 0))
                total = s + a + b
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 監控狀態:")
                print(f"  S層: {s} 檔 | " f"A層: {a} 檔 | " f"B層: {b} 檔")
                print(
                    f"  總股票數: {total} | "
                    f"運行中: {'是' if status.get('is_running', False) else '否'}"
                )

                # 獲取各層詳細信息
                tier_details = monitor.get_tier_details()
                if "S_tier" in tier_details and tier_details["S_tier"]["stocks"]:
                    print(f"\n  S層熱門股票: {', '.join(tier_details['S_tier']['stocks'][:5])}")

                # 顯示最新信號（如果有）
                for tier_level in [TierLevel.S_TIER, TierLevel.A_TIER]:
                    tier_info = monitor.get_tier_details(tier_level)
                    if tier_info and tier_info.get(tier_level.value, {}).get("recent_signals"):
                        for signal in tier_info[tier_level.value]["recent_signals"][:3]:
                            print(
                                f"  💎 {signal['symbol']}: {signal['type']} (強度: {signal.get('strength', 0):.2f})"
                            )

            # 等待1秒
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\n[!] 用戶中斷，正在關閉系統...")
    except Exception as e:
        print(f"\n[ERROR] 系統錯誤: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n" + "=" * 80)
        print("系統已停止")
        print("=" * 80)


def create_full_stock_list():
    """創建完整的股票列表文件（可選）"""
    # 這裡可以從各種來源獲取股票列表
    # 例如：從Yahoo Finance獲取所有美股

    import yfinance as yf
    import pandas as pd

    print("獲取股票列表...")

    # 獲取不同市場的股票
    # 這是示範，實際可以從更多來源獲取

    # NASDAQ 100
    nasdaq100 = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "PYPL",
        "ADBE",
        "NFLX",
        "CMCSA",
        "CSCO",
        "INTC",
        "PEP",
        "AVGO",
        "TXN",
        "QCOM",
        "COST",
        "TMUS",
        "CHTR",
    ]

    # 將列表擴展到4000+（這裡用重複作為示範）
    all_symbols = []
    base_symbols = nasdaq100

    # 生成4000個符號（實際應該從真實數據源獲取）
    for i in range(200):  # 200 * 20 = 4000
        for symbol in base_symbols:
            if i == 0:
                all_symbols.append(symbol)
            else:
                # 添加數字後綴作為示範（實際應該是真實股票代碼）
                all_symbols.append(f"{symbol}.{i}")

    # 保存到文件
    with open("data/all_symbols.txt", "w") as f:
        for symbol in all_symbols[:4000]:  # 限制為4000個
            f.write(f"{symbol}\n")

    print(f"已保存 {len(all_symbols[:4000])} 個股票符號")


if __name__ == "__main__":
    # 檢查Python版本
    if sys.version_info < (3, 7):
        print("ERROR: 需要 Python 3.7+")
        sys.exit(1)

    # 檢查必要的套件
    try:
        import pandas
        import numpy
        import yfinance
        import zmq  # 檢查ZeroMQ
    except ImportError as e:
        print(f"ERROR: 缺少必要套件: {e}")
        print("請運行: pip install pandas numpy yfinance pyzmq python-dotenv")
        sys.exit(1)

    # 選項：創建股票列表
    import argparse

    parser = argparse.ArgumentParser(description="4000+股票監控系統")
    parser.add_argument("--create-list", action="store_true", help="創建4000股票列表文件")
    args = parser.parse_args()

    if args.create_list:
        create_full_stock_list()
    else:
        # 運行主程序
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n系統關閉完成")
        except Exception as e:
            print(f"致命錯誤: {e}")
            sys.exit(1)
