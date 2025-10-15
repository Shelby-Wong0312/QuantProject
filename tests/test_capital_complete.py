# test_capital_complete.py
# 完整的Capital.com API測試程序

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from tabulate import tabulate

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# 關閉config的調試輸出
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.stdout = open(os.devnull, "w")
import config

sys.stdout = sys.__stdout__

# 導入數據模組
from data_pipeline.capital_history_loader import CapitalHistoryLoader
from data_pipeline.data_manager import DataManager


def print_section(title):
    """打印分節標題"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")


def test_api_login():
    """測試API登錄功能"""
    print_section("測試1: API登錄")
    try:
        loader = CapitalHistoryLoader()
        if loader.cst and loader.x_security_token:
            print("✅ API登錄成功")
            print(f"CST Token: {loader.cst[:20]}...")
            print(f"Security Token: {loader.x_security_token[:20]}...")
            return loader
        else:
            print("❌ API登錄失敗")
            return None
    except Exception as e:
        print(f"❌ 登錄時發生錯誤: {e}")
        return None


def test_get_available_symbols(loader):
    """測試獲取可用交易品種"""
    print_section("測試2: 獲取可用交易品種")
    try:
        symbols = loader.get_available_symbols()
        if symbols:
            print(f"✅ 成功獲取 {len(symbols)} 個可用交易品種")
            print("\n前10個品種:")
            for i, symbol in enumerate(symbols[:10]):
                print(f"  {i+1}. {symbol}")

            # 搜索特定股票
            us_stocks = [s for s in symbols if ".US" in s]
            print(f"\n美股數量: {len(us_stocks)}")
            if us_stocks:
                print("部分美股代碼:")
                for stock in us_stocks[:5]:
                    print(f"  - {stock}")
        else:
            print("❌ 未能獲取交易品種列表")
        return symbols
    except Exception as e:
        print(f"❌ 獲取交易品種時發生錯誤: {e}")
        return []


def test_get_historical_data(loader, symbol="AAPL.US"):
    """測試獲取歷史K線數據"""
    print_section(f"測試3: 獲取歷史K線數據 ({symbol})")

    # 測試不同時間週期
    resolutions = {
        "MINUTE": "1分鐘",
        "MINUTE_5": "5分鐘",
        "MINUTE_15": "15分鐘",
        "HOUR": "1小時",
        "DAY": "日線",
    }

    end_date = datetime.now()
    results = []

    for res_code, res_name in resolutions.items():
        try:
            # 根據時間週期調整查詢範圍
            if "MINUTE" in res_code:
                start_date = end_date - timedelta(days=1)
            elif res_code == "HOUR":
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=30)

            print(f"\n正在獲取{res_name}數據...")
            df = loader.get_bars(
                symbol=symbol,
                resolution=res_code,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                max_results=100,
            )

            if not df.empty:
                results.append(
                    {
                        "時間週期": res_name,
                        "數據筆數": len(df),
                        "開始時間": df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                        "結束時間": df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                        "最高價": f"${df['High'].max():.2f}",
                        "最低價": f"${df['Low'].min():.2f}",
                    }
                )
                print(f"✅ 成功獲取 {len(df)} 筆數據")

                # 顯示最新5筆數據
                print(f"\n最新5筆{res_name}數據:")
                latest_data = df.tail(5)
                print(
                    tabulate(
                        latest_data,
                        headers=["日期時間", "開盤", "最高", "最低", "收盤", "成交量"],
                        floatfmt=".2f",
                        tablefmt="grid",
                    )
                )
            else:
                print(f"❌ 未能獲取{res_name}數據")

            time.sleep(1)  # 避免API請求過快

        except Exception as e:
            print(f"❌ 獲取{res_name}數據時發生錯誤: {e}")

    # 顯示匯總結果
    if results:
        print("\n📊 數據獲取匯總:")
        print(tabulate(results, headers="keys", tablefmt="pretty"))

    return results


def test_get_realtime_price(loader, symbols=["AAPL.US", "MSFT.US", "GOOGL.US"]):
    """測試獲取實時價格"""
    print_section("測試4: 獲取實時價格")

    prices = []
    for symbol in symbols:
        try:
            market_info = loader.get_market_info(symbol)
            if market_info and "snapshot" in market_info:
                snapshot = market_info["snapshot"]
                if snapshot.get("offer") and snapshot.get("bid"):
                    bid = snapshot["bid"]
                    ask = snapshot["offer"]
                    mid = (bid + ask) / 2
                    spread = ask - bid

                    prices.append(
                        {
                            "股票代碼": symbol,
                            "買價": f"${bid:.2f}",
                            "賣價": f"${ask:.2f}",
                            "中間價": f"${mid:.2f}",
                            "價差": f"${spread:.4f}",
                            "價差%": f"{(spread/mid)*100:.3f}%",
                        }
                    )
                    print(f"✅ {symbol}: 中間價 ${mid:.2f}")
            else:
                print(f"❌ 無法獲取 {symbol} 的實時價格")

            time.sleep(0.5)
        except Exception as e:
            print(f"❌ 獲取 {symbol} 價格時發生錯誤: {e}")

    if prices:
        print("\n實時價格明細:")
        print(tabulate(prices, headers="keys", tablefmt="pretty"))

    return prices


def test_data_cache(symbol="AAPL.US"):
    """測試數據緩存功能"""
    print_section("測試5: 數據緩存功能")

    try:
        # 創建數據管理器（啟用緩存）
        dm = DataManager(use_cache=True)

        # 第一次獲取數據（從API）
        print(f"第一次獲取 {symbol} 數據（從API）...")
        start_time = time.time()
        df1 = dm.get_historical_data(symbol, resolution="DAY", lookback_days=30)
        api_time = time.time() - start_time
        print(f"✅ 從API獲取成功，耗時: {api_time:.2f}秒，數據筆數: {len(df1)}")

        # 第二次獲取數據（從緩存）
        print(f"\n第二次獲取 {symbol} 數據（從緩存）...")
        start_time = time.time()
        df2 = dm.get_historical_data(symbol, resolution="DAY", lookback_days=30)
        cache_time = time.time() - start_time
        print(f"✅ 從緩存獲取成功，耗時: {cache_time:.2f}秒，數據筆數: {len(df2)}")

        # 比較性能
        speedup = api_time / cache_time if cache_time > 0 else float("inf")
        print(f"\n性能提升: {speedup:.1f}倍")

        # 顯示緩存統計
        stats = dm.get_cache_stats()
        if stats:
            print(f"\n緩存統計:")
            print(f"  - 緩存條目數: {stats.get('total_entries', 0)}")
            print(f"  - 緩存大小: {stats.get('cache_size_mb', 0):.2f} MB")
            print(f"  - 最舊數據: {stats.get('oldest_entry', 'N/A')}")
            print(f"  - 最新數據: {stats.get('newest_entry', 'N/A')}")

        dm.close()
        return True

    except Exception as e:
        print(f"❌ 測試緩存功能時發生錯誤: {e}")
        return False


def test_batch_data_fetch():
    """測試批量數據獲取"""
    print_section("測試6: 批量數據獲取")

    symbols = ["AAPL.US", "MSFT.US", "GOOGL.US", "TSLA.US", "AMZN.US"]

    try:
        dm = DataManager(use_cache=True)

        print(f"正在批量獲取 {len(symbols)} 隻股票的數據...")
        start_time = time.time()

        # 批量獲取歷史數據
        results = dm.get_multiple_symbols_data(
            symbols=symbols, resolution="DAY", lookback_days=30, max_workers=3
        )

        elapsed_time = time.time() - start_time

        # 統計結果
        success_count = sum(1 for df in results.values() if not df.empty)
        print(f"\n✅ 批量獲取完成，耗時: {elapsed_time:.2f}秒")
        print(f"成功率: {success_count}/{len(symbols)} ({success_count/len(symbols)*100:.1f}%)")

        # 顯示各股票數據情況
        summary = []
        for symbol, df in results.items():
            if not df.empty:
                summary.append(
                    {
                        "股票代碼": symbol,
                        "數據筆數": len(df),
                        "開始日期": df.index[0].strftime("%Y-%m-%d"),
                        "結束日期": df.index[-1].strftime("%Y-%m-%d"),
                        "最新收盤價": f"${df['Close'].iloc[-1]:.2f}",
                    }
                )

        if summary:
            print("\n批量獲取結果:")
            print(tabulate(summary, headers="keys", tablefmt="pretty"))

        # 批量獲取最新價格
        print(f"\n正在批量獲取最新價格...")
        prices = dm.get_batch_latest_prices(symbols)

        if prices:
            price_table = []
            for symbol, price in prices.items():
                price_table.append({"股票代碼": symbol, "最新價格": f"${price:.2f}"})

            print("\n批量價格結果:")
            print(tabulate(price_table, headers="keys", tablefmt="pretty"))

        dm.close()
        return True

    except Exception as e:
        print(f"❌ 批量獲取數據時發生錯誤: {e}")
        return False


def main():
    """主測試函數"""
    print("\n")
    print("=" * 60)
    print("Capital.com API 完整功能測試")
    print("=" * 60)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 測試1: API登錄
    loader = test_api_login()
    if not loader:
        print("\n❌ API登錄失敗，無法繼續測試")
        return

    # 測試2: 獲取可用交易品種
    symbols = test_get_available_symbols(loader)

    # 測試3: 獲取歷史K線數據
    test_get_historical_data(loader)

    # 測試4: 獲取實時價格
    test_get_realtime_price(loader)

    # 關閉loader
    loader.close()

    # 測試5: 數據緩存功能
    test_data_cache()

    # 測試6: 批量數據獲取
    test_batch_data_fetch()

    print("\n")
    print("=" * 60)
    print("所有測試完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n測試被用戶中斷")
    except Exception as e:
        print(f"\n\n測試過程中發生未預期的錯誤: {e}")
        import traceback

        traceback.print_exc()
