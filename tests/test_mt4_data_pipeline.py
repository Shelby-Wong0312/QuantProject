#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4數據管道測試腳本
測試數據收集、質量檢查、緩存和統一接口功能
"""

import unittest
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.mt4_data_collector import (
    MT4DataPipeline,
    MarketData,
    DataQualityChecker,
    DataCache,
    get_pipeline,
    create_pipeline,
    start_data_collection,
    get_realtime_data,
    get_historical_data,
)
from mt4_bridge.data_collector import TickData, OHLCData, TimeFrame


class TestDataQualityChecker(unittest.TestCase):
    """測試數據質量檢查器"""

    def setUp(self):
        self.checker = DataQualityChecker()

    def test_valid_tick(self):
        """測試有效的Tick數據"""
        tick = TickData(
            symbol="EURUSD", timestamp=datetime.now(), bid=1.0800, ask=1.0802, spread=2, volume=100
        )
        is_valid, error = self.checker.check_tick(tick)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_invalid_spread(self):
        """測試點差過大的Tick"""
        tick = TickData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            bid=1.0800,
            ask=1.1000,  # 點差太大
            spread=200,
            volume=100,
        )
        is_valid, error = self.checker.check_tick(tick)
        self.assertFalse(is_valid)
        self.assertIn("Spread too wide", error)

    def test_price_jump(self):
        """測試價格跳動檢查"""
        # 第一個tick
        tick1 = TickData(
            symbol="EURUSD", timestamp=datetime.now(), bid=1.0800, ask=1.0802, spread=2, volume=100
        )
        self.checker.check_tick(tick1)

        # 價格跳動過大的tick
        tick2 = TickData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            bid=1.2000,  # 跳動過大
            ask=1.2002,
            spread=2,
            volume=100,
        )
        is_valid, error = self.checker.check_tick(tick2)
        self.assertFalse(is_valid)
        self.assertIn("Price jump too large", error)

    def test_valid_ohlc(self):
        """測試有效的OHLC數據"""
        ohlc = OHLCData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            timeframe=TimeFrame.M5,
            open=1.0800,
            high=1.0850,
            low=1.0790,
            close=1.0820,
            volume=1000,
            tick_count=50,
        )
        is_valid, error = self.checker.check_ohlc(ohlc)
        self.assertTrue(is_valid)

    def test_invalid_ohlc(self):
        """測試無效的OHLC數據"""
        # High < Low
        ohlc = OHLCData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            timeframe=TimeFrame.M5,
            open=1.0800,
            high=1.0790,  # 低於low
            low=1.0850,
            close=1.0820,
            volume=1000,
            tick_count=50,
        )
        is_valid, error = self.checker.check_ohlc(ohlc)
        self.assertFalse(is_valid)
        self.assertIn("High < Low", error)


class TestDataCache(unittest.TestCase):
    """測試數據緩存"""

    def setUp(self):
        self.cache = DataCache(max_tick_cache=100, max_ohlc_cache=50)

    def test_tick_cache(self):
        """測試Tick緩存"""
        # 添加tick數據
        for i in range(10):
            tick = TickData(
                symbol="EURUSD",
                timestamp=datetime.now() + timedelta(seconds=i),
                bid=1.0800 + i * 0.0001,
                ask=1.0802 + i * 0.0001,
                spread=2,
                volume=100 + i,
            )
            self.cache.add_tick(tick)

        # 獲取最近的tick
        recent_ticks = self.cache.get_recent_ticks("EURUSD", 5)
        self.assertEqual(len(recent_ticks), 5)

        # 檢查順序
        for i in range(1, len(recent_ticks)):
            self.assertGreater(recent_ticks[i].timestamp, recent_ticks[i - 1].timestamp)

    def test_ohlc_cache(self):
        """測試OHLC緩存"""
        # 添加OHLC數據
        for i in range(10):
            ohlc = OHLCData(
                symbol="EURUSD",
                timestamp=datetime.now() + timedelta(minutes=i * 5),
                timeframe=TimeFrame.M5,
                open=1.0800,
                high=1.0850,
                low=1.0790,
                close=1.0820 + i * 0.0001,
                volume=1000 + i * 10,
                tick_count=50,
            )
            self.cache.add_ohlc(ohlc)

        # 獲取最近的OHLC
        recent_bars = self.cache.get_recent_ohlc("EURUSD", TimeFrame.M5, 5)
        self.assertEqual(len(recent_bars), 5)

    def test_to_dataframe(self):
        """測試轉換為DataFrame"""
        # 添加tick數據
        for i in range(5):
            tick = TickData(
                symbol="GBPUSD",
                timestamp=datetime.now() + timedelta(seconds=i),
                bid=1.2500 + i * 0.0001,
                ask=1.2502 + i * 0.0001,
                spread=2,
                volume=100,
            )
            self.cache.add_tick(tick)

        # 轉換為DataFrame
        df = self.cache.to_dataframe("GBPUSD")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn("bid", df.columns)
        self.assertIn("ask", df.columns)


class TestMT4DataPipeline(unittest.TestCase):
    """測試MT4數據管道"""

    def setUp(self):
        # 創建測試管道（不連接真實MT4）
        self.pipeline = MT4DataPipeline(
            connector=None,
            enable_storage=False,  # 測試時不使用存儲
            enable_cache=True,
            enable_quality_check=True,
        )

    def test_market_data_creation(self):
        """測試市場數據創建"""
        tick = TickData(
            symbol="EURUSD", timestamp=datetime.now(), bid=1.0800, ask=1.0802, spread=2, volume=100
        )

        market_data = self.pipeline._create_market_data(tick)

        self.assertIsInstance(market_data, MarketData)
        self.assertEqual(market_data.symbol, "EURUSD")
        self.assertEqual(market_data.bid, 1.0800)
        self.assertEqual(market_data.ask, 1.0802)
        self.assertEqual(market_data.mid, 1.0801)
        self.assertEqual(market_data.spread, 2)

    def test_subscribe_unsubscribe(self):
        """測試訂閱和取消訂閱"""
        # 訂閱
        self.pipeline.subscribe(["EURUSD", "GBPUSD"])
        self.assertIn("EURUSD", self.pipeline.subscribed_symbols)
        self.assertIn("GBPUSD", self.pipeline.subscribed_symbols)

        # 取消訂閱
        self.pipeline.unsubscribe("EURUSD")
        self.assertNotIn("EURUSD", self.pipeline.subscribed_symbols)
        self.assertIn("GBPUSD", self.pipeline.subscribed_symbols)

    def test_callback_system(self):
        """測試回調系統"""
        received_data = []

        def callback(data: MarketData):
            received_data.append(data)

        # 添加回調
        self.pipeline.add_callback(callback)

        # 模擬tick數據
        tick = TickData(
            symbol="EURUSD", timestamp=datetime.now(), bid=1.0800, ask=1.0802, spread=2, volume=100
        )

        # 處理tick（會觸發回調）
        self.pipeline._process_tick(tick)

        # 檢查回調是否被觸發
        self.assertEqual(len(received_data), 1)
        self.assertEqual(received_data[0].symbol, "EURUSD")

    def test_statistics(self):
        """測試統計功能"""
        # 模擬處理一些tick
        for i in range(10):
            tick = TickData(
                symbol="EURUSD",
                timestamp=datetime.now(),
                bid=1.0800 + i * 0.0001,
                ask=1.0802 + i * 0.0001,
                spread=2,
                volume=100,
            )
            self.pipeline._process_tick(tick)

        # 獲取統計
        stats = self.pipeline.get_stats()

        self.assertEqual(stats["total_ticks"], 10)
        self.assertEqual(stats["valid_ticks"], 10)
        self.assertEqual(stats["invalid_ticks"], 0)
        self.assertGreater(stats["validity_rate"], 0.99)

    def test_get_indicators(self):
        """測試指標計算"""
        # 添加足夠的OHLC數據
        for i in range(100):
            ohlc = OHLCData(
                symbol="EURUSD",
                timestamp=datetime.now() - timedelta(minutes=(99 - i) * 5),
                timeframe=TimeFrame.M5,
                open=1.0800 + i * 0.0001,
                high=1.0850 + i * 0.0001,
                low=1.0790 + i * 0.0001,
                close=1.0820 + i * 0.0001,
                volume=1000,
                tick_count=50,
            )
            if self.pipeline.cache:
                self.pipeline.cache.add_ohlc(ohlc)

        # 計算指標
        indicators = self.pipeline.get_indicators("EURUSD", TimeFrame.M5)

        # 檢查SMA20
        if "sma20" in indicators:
            self.assertIsInstance(indicators["sma20"], float)
            self.assertGreater(indicators["sma20"], 0)

        # 檢查RSI
        if "rsi14" in indicators:
            self.assertIsInstance(indicators["rsi14"], float)
            self.assertGreaterEqual(indicators["rsi14"], 0)
            self.assertLessEqual(indicators["rsi14"], 100)


class TestGlobalFunctions(unittest.TestCase):
    """測試全局函數"""

    def test_get_pipeline_singleton(self):
        """測試獲取單例管道"""
        pipeline1 = get_pipeline()
        pipeline2 = get_pipeline()
        self.assertIs(pipeline1, pipeline2)

    def test_create_pipeline(self):
        """測試創建新管道"""
        pipeline1 = create_pipeline()
        pipeline2 = create_pipeline()
        self.assertIsNot(pipeline1, pipeline2)


def run_integration_test():
    """運行集成測試（需要MT4連接）"""
    print("\n" + "=" * 60)
    print("MT4數據管道集成測試")
    print("=" * 60)

    # 創建管道
    pipeline = create_pipeline()

    # 測試回調
    def on_data(data: MarketData):
        print(f"\n收到數據: {data.symbol}")
        print(f"  時間: {data.timestamp}")
        print(f"  Bid: {data.bid}, Ask: {data.ask}")
        print(f"  Mid: {data.mid}, Spread: {data.spread}")
        if data.indicators:
            print(f"  指標: {data.indicators}")

    pipeline.add_callback(on_data)

    # 嘗試連接
    print("\n正在連接MT4...")
    if pipeline.connect():
        print("✓ 連接成功")

        # 訂閱品種
        pipeline.subscribe("EURUSD")
        print("✓ 已訂閱EURUSD")

        # 啟動數據收集
        pipeline.start()
        print("✓ 數據收集已啟動")

        # 運行5秒
        print("\n收集數據中（5秒）...")
        time.sleep(5)

        # 獲取統計
        stats = pipeline.get_stats()
        print("\n統計信息:")
        print(f"  總Tick數: {stats['total_ticks']}")
        print(f"  有效Tick數: {stats['valid_ticks']}")
        print(f"  無效Tick數: {stats['invalid_ticks']}")
        print(f"  有效率: {stats['validity_rate']:.2%}")

        # 獲取最新數據
        latest = pipeline.get_latest_data("EURUSD")
        if latest:
            print("\n最新數據:")
            print(f"  Bid: {latest.bid}, Ask: {latest.ask}")

        # 獲取歷史數據
        df = pipeline.get_dataframe("EURUSD", TimeFrame.M5, 10)
        if not df.empty:
            print("\n歷史數據（最近10條）:")
            print(df.tail(5))

        # 停止
        pipeline.stop()
        print("\n✓ 數據收集已停止")

        # 斷開連接
        pipeline.disconnect()
        print("✓ 已斷開連接")

    else:
        print("✗ 無法連接到MT4")
        print("  請確保MT4已啟動並載入PythonBridge EA")


if __name__ == "__main__":
    # 運行單元測試
    print("運行單元測試...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # 詢問是否運行集成測試
    print("\n" + "=" * 60)
    response = input("是否運行集成測試（需要MT4連接）? (y/n): ")
    if response.lower() == "y":
        run_integration_test()
    else:
        print("跳過集成測試")
