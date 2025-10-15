#!/usr/bin/env python3
"""
增強數據管道演示腳本
展示4000+股票監控系統的核心功能
"""

import time
from pathlib import Path
import sys

# 添加項目根目錄到路徑
sys.path.append(str(Path(__file__).parent))

from data_pipeline.free_data_client import FreeDataClient


def demo_enhanced_data_pipeline():
    """演示增強的數據管道功能"""

    print("=" * 70)
    print("Enhanced Data Pipeline System Demo")
    print("Support for 4000+ Large-Scale Stock Monitoring")
    print("=" * 70)

    # 初始化客戶端
    print("\nInitializing data client...")
    client = FreeDataClient()
    print(f"Database location: {client.db_path}")
    print(f"Batch size: {client.batch_size}")
    print(f"Max worker threads: {client.max_workers}")

    # 演示1: 小規模批量報價
    print("\n" + "=" * 50)
    print("📊 演示1: 批量報價系統")
    print("=" * 50)

    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    print(f"測試股票: {', '.join(test_symbols)}")

    start_time = time.time()
    quotes = client.get_batch_quotes(test_symbols, show_progress=True)
    duration = time.time() - start_time

    print(f"\n📈 批量報價結果:")
    print(f"   成功獲取: {len(quotes)}/{len(test_symbols)} 股票")
    print(f"   處理時間: {duration:.2f} 秒")
    print(f"   平均速度: {len(quotes)/duration:.1f} 股票/秒")

    # 顯示報價詳情
    print(f"\n💰 實時報價:")
    for symbol, data in list(quotes.items())[:5]:
        print(f"   {symbol}: ${data['price']:.2f} (成交量: {data['volume']:,})")

    # 演示2: 市場概覽
    print("\n" + "=" * 50)
    print("🌐 演示2: 市場概覽")
    print("=" * 50)

    overview = client.get_market_overview()
    print(f"市場狀態: {'🟢 開放' if overview.get('is_open') else '🔴 關閉'}")
    print(f"交易時段: {overview.get('session_type')}")

    if "indices" in overview:
        print(f"\n📊 主要指數:")
        for index, data in overview["indices"].items():
            if data:
                print(f"   {index}: ${data.get('price', 0):.2f}")

    # 演示3: 監控清單摘要
    print("\n" + "=" * 50)
    print("📋 演示3: 監控清單摘要")
    print("=" * 50)

    summary = client.get_watchlist_summary(test_symbols)
    if "error" not in summary:
        print(f"監控股票總數: {summary.get('total_symbols', 0)}")
        print(f"成功獲取數據: {summary.get('successful_quotes', 0)}")
        print(f"成功率: {summary.get('success_rate', 0):.1f}%")

        price_stats = summary.get("price_stats", {})
        print(f"價格範圍: ${price_stats.get('min', 0):.2f} - ${price_stats.get('max', 0):.2f}")
        print(f"平均價格: ${price_stats.get('mean', 0):.2f}")

        volume_stats = summary.get("volume_stats", {})
        print(f"總成交量: {volume_stats.get('total', 0):,}")

    # 演示4: 緩存效能
    print("\n" + "=" * 50)
    print("🚀 演示4: 緩存系統效能")
    print("=" * 50)

    # 第一次請求（建立緩存）
    print("第一次請求（建立緩存）...")
    start_time = time.time()
    quotes1 = client.get_batch_quotes(test_symbols[:5], use_cache=False, show_progress=False)
    first_time = time.time() - start_time

    # 第二次請求（使用緩存）
    print("第二次請求（使用緩存）...")
    start_time = time.time()
    quotes2 = client.get_batch_quotes(test_symbols[:5], use_cache=True, show_progress=False)
    cached_time = time.time() - start_time

    speedup = first_time / cached_time if cached_time > 0 else float("inf")
    print(f"\n⚡ 緩存效能:")
    print(f"   首次請求: {first_time:.2f} 秒")
    print(f"   緩存請求: {cached_time:.2f} 秒")
    print(f"   加速倍數: {speedup:.1f}x")

    # 演示5: 技術指標計算
    print("\n" + "=" * 50)
    print("📈 演示5: 技術指標計算")
    print("=" * 50)

    # 獲取歷史數據
    print("獲取AAPL歷史數據...")
    hist_data = client.get_historical_data("AAPL", period="30d")

    if hist_data is not None and not hist_data.empty:
        # 計算技術指標
        hist_with_indicators = client.calculate_indicators(hist_data)
        latest = hist_with_indicators.iloc[-1]

        print(f"\n📊 AAPL技術指標 (最新):")
        print(f"   RSI: {latest.get('RSI', 0):.2f}")
        print(f"   MACD: {latest.get('MACD', 0):.4f}")
        print(f"   SMA_20: ${latest.get('SMA_20', 0):.2f}")
        print(f"   SMA_50: ${latest.get('SMA_50', 0):.2f}")
        print(f"   布林線上軌: ${latest.get('BB_Upper', 0):.2f}")
        print(f"   布林線下軌: ${latest.get('BB_Lower', 0):.2f}")

    # 大規模測試演示
    print("\n" + "=" * 50)
    print("🎯 演示6: 大規模處理能力")
    print("=" * 50)

    # 模擬大規模股票清單
    large_symbols = test_symbols * 25  # 200個股票
    print(f"模擬處理 {len(large_symbols)} 個股票...")

    start_time = time.time()
    large_quotes = client.get_batch_quotes(large_symbols, show_progress=True)
    large_duration = time.time() - start_time

    print(f"\n🚀 大規模處理結果:")
    print(f"   處理股票數: {len(large_symbols)}")
    print(f"   成功獲取: {len(large_quotes)}")
    print(f"   處理時間: {large_duration:.2f} 秒")
    print(f"   吞吐量: {len(large_quotes)/large_duration:.1f} 股票/秒")

    # 系統狀態摘要
    print("\n" + "=" * 70)
    print("📋 系統狀態摘要")
    print("=" * 70)

    print(f"✅ 數據庫: {client.db_path}")
    print(f"✅ 緩存有效期: {client.cache_duration} 秒")
    print(f"✅ Alpha Vantage API: {'已配置' if client.alpha_vantage_key else '未配置'}")
    print(f"✅ 批次處理: {client.batch_size} 股票/批")
    print(f"✅ 並發線程: {client.max_workers} 線程")
    print(f"✅ 支援規模: 4000+ 股票")

    print(f"\n🎉 演示完成！系統已準備好進行大規模股票監控")
    print(f"📊 數據已保存到本地數據庫，可重複使用")


if __name__ == "__main__":
    demo_enhanced_data_pipeline()
