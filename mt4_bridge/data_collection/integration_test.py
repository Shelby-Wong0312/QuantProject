# -*- coding: utf-8 -*-
"""
MT4 數據收集系統整合測試
測試與現有 EventLoop 和系統的兼容性
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

try:
    # 導入核心事件系統
    from core.event import EventType, MarketEvent
    from core.event_loop import EventLoop
    
    # 導入 MT4 數據收集組件
    from mt4_bridge.data_collection import (
        TickCollector, TickData, 
        OHLCAggregator, OHLCBar, TimeFrame,
        MT4DataFeed, DataStorage
    )
    
    # 導入錯誤處理
    from mt4_bridge.data_collection.error_handler import (
        setup_logging, LogLevel, handle_error, 
        TickCollectionError, DataFeedError
    )
    
    IMPORTS_SUCCESS = True
    
except ImportError as e:
    print(f"導入失敗: {e}")
    IMPORTS_SUCCESS = False

class MockMT4Bridge:
    """模擬的 MT4 橋接器，用於測試"""
    
    def __init__(self):
        self.is_connected = False
        self.tick_counter = 0
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
    def get_account_info(self):
        """模擬獲取帳戶信息"""
        if not self.is_connected:
            # 模擬連接
            self.is_connected = True
            
        return {
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0
        }
    
    def send_command(self, command, **kwargs):
        """模擬發送命令"""
        pass
    
    def receive_data(self, timeout=1000):
        """模擬接收數據"""
        import random
        import time
        
        # 模擬沒有數據的情況
        if random.random() < 0.3:
            return None
            
        # 生成模擬 Tick 數據
        symbol = random.choice(self.symbols)
        base_price = {"EURUSD": 1.1000, "GBPUSD": 1.3000, "USDJPY": 110.00}[symbol]
        
        # 添加隨機波動
        price_change = random.uniform(-0.001, 0.001)
        current_price = base_price + price_change
        
        spread = {"EURUSD": 0.0001, "GBPUSD": 0.0002, "USDJPY": 0.01}[symbol]
        
        self.tick_counter += 1
        
        return {
            "symbol": symbol,
            "bid": current_price - spread/2,
            "ask": current_price + spread/2,
            "last": current_price,
            "volume": random.randint(1, 100),
            "timestamp": time.time()
        }
    
    def close(self):
        """關閉連接"""
        self.is_connected = False

class IntegrationTester:
    """整合測試器"""
    
    def __init__(self):
        self.logger = setup_logging(LogLevel.INFO, console_output=True)
        self.test_results = {}
        self.mock_bridge = MockMT4Bridge()
    
    async def test_event_system_compatibility(self):
        """測試事件系統兼容性"""
        test_name = "event_system_compatibility"
        self.logger.info(f"開始測試: {test_name}")
        
        try:
            # 創建事件循環
            event_loop = EventLoop()
            
            # 創建事件處理器
            received_events = []
            
            async def market_event_handler(event):
                received_events.append(event)
                self.logger.info(f"收到市場事件: {event.symbol} @{event.timestamp}")
            
            # 註冊處理器
            event_loop.add_handler(EventType.MARKET, market_event_handler)
            
            # 手動創建並發送測試事件
            test_event = MarketEvent(
                symbol="EURUSD",
                timestamp=datetime.now(timezone.utc),
                ohlcv_data=self._create_test_dataframe()
            )
            
            # 啟動事件循環
            event_task = asyncio.create_task(event_loop.run())
            
            # 發送測試事件
            await event_loop.put_event(test_event)
            
            # 等待事件處理
            await asyncio.sleep(0.1)
            
            # 停止事件循環
            event_loop.stop()
            event_task.cancel()
            
            try:
                await event_task
            except asyncio.CancelledError:
                pass
            
            # 驗證結果
            success = len(received_events) > 0
            
            self.test_results[test_name] = {
                "success": success,
                "events_received": len(received_events),
                "message": "事件系統兼容性測試通過" if success else "事件系統兼容性測試失敗"
            }
            
            self.logger.info(f"測試 {test_name} 結果: {'成功' if success else '失敗'}")
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "error": str(e),
                "message": f"事件系統兼容性測試出錯: {e}"
            }
            handle_error(e, f"測試 {test_name}")
    
    async def test_tick_collector_basic(self):
        """測試 Tick 收集器基本功能"""
        test_name = "tick_collector_basic"
        self.logger.info(f"開始測試: {test_name}")
        
        try:
            # 創建 Tick 收集器
            collector = TickCollector(
                symbols=["EURUSD", "GBPUSD"],
                mt4_bridge=self.mock_bridge,
                cache_size=100
            )
            
            # 測試手動添加 Tick
            test_tick = TickData(
                symbol="EURUSD",
                timestamp=datetime.now(timezone.utc),
                bid=1.1000,
                ask=1.1002
            )
            
            # 創建回調來收集 Tick
            received_ticks = []
            def tick_callback(tick):
                received_ticks.append(tick)
            
            collector.add_callback(tick_callback)
            
            # 模擬處理 Tick
            collector._store_tick(test_tick)
            
            # 驗證結果
            latest_tick = collector.get_latest_tick("EURUSD")
            recent_ticks = collector.get_recent_ticks("EURUSD", 10)
            
            success = (
                latest_tick is not None and
                len(recent_ticks) > 0 and
                len(received_ticks) > 0
            )
            
            self.test_results[test_name] = {
                "success": success,
                "latest_tick": latest_tick.symbol if latest_tick else None,
                "recent_ticks_count": len(recent_ticks),
                "callbacks_triggered": len(received_ticks),
                "message": "Tick 收集器基本功能測試通過" if success else "Tick 收集器基本功能測試失敗"
            }
            
            self.logger.info(f"測試 {test_name} 結果: {'成功' if success else '失敗'}")
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "error": str(e),
                "message": f"Tick 收集器基本功能測試出錯: {e}"
            }
            handle_error(e, f"測試 {test_name}")
    
    async def test_ohlc_aggregator_basic(self):
        """測試 OHLC 聚合器基本功能"""
        test_name = "ohlc_aggregator_basic"
        self.logger.info(f"開始測試: {test_name}")
        
        try:
            # 創建 OHLC 聚合器
            aggregator = OHLCAggregator(
                symbols=["EURUSD"],
                timeframes=[TimeFrame.M1, TimeFrame.M5],
                enable_indicators=False  # 暫時關閉指標以簡化測試
            )
            
            # 創建回調來收集完成的 K 線
            completed_bars = []
            def bar_callback(bar):
                completed_bars.append(bar)
            
            aggregator.add_bar_callback(bar_callback)
            
            # 生成一系列 Tick 數據
            base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            
            for i in range(5):
                tick = TickData(
                    symbol="EURUSD",
                    timestamp=base_time + timedelta(seconds=i * 10),
                    bid=1.1000 + i * 0.0001,
                    ask=1.1002 + i * 0.0001
                )
                aggregator.process_tick(tick)
            
            # 等待處理
            await asyncio.sleep(0.1)
            
            # 檢查當前 K 線
            current_bar = aggregator.get_current_bar("EURUSD", TimeFrame.M1)
            
            success = current_bar is not None
            
            self.test_results[test_name] = {
                "success": success,
                "current_bar_symbol": current_bar.symbol if current_bar else None,
                "completed_bars": len(completed_bars),
                "message": "OHLC 聚合器基本功能測試通過" if success else "OHLC 聚合器基本功能測試失敗"
            }
            
            self.logger.info(f"測試 {test_name} 結果: {'成功' if success else '失敗'}")
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "error": str(e),
                "message": f"OHLC 聚合器基本功能測試出錯: {e}"
            }
            handle_error(e, f"測試 {test_name}")
    
    async def test_data_storage_basic(self):
        """測試數據存儲基本功能"""
        test_name = "data_storage_basic"
        self.logger.info(f"開始測試: {test_name}")
        
        try:
            # 創建測試存儲路徑
            test_storage_path = "./test_storage"
            
            # 創建數據存儲系統
            storage = DataStorage(
                storage_path=test_storage_path,
                enable_sqlite=True,
                enable_parquet=False,  # 暫時只測試 SQLite
                enable_csv=False,
                batch_size=10
            )
            
            storage.start()
            
            # 創建測試 Tick 數據
            test_tick = TickData(
                symbol="EURUSD",
                timestamp=datetime.now(timezone.utc),
                bid=1.1000,
                ask=1.1002
            )
            
            # 存儲 Tick 數據
            storage.store_tick(test_tick)
            
            # 刷新緩存
            storage.flush_cache()
            
            # 等待寫入完成
            await asyncio.sleep(1)
            
            # 查詢數據
            retrieved_ticks = storage.query_tick_data("EURUSD", limit=10)
            
            # 停止存儲系統
            storage.stop()
            
            success = len(retrieved_ticks) > 0
            
            self.test_results[test_name] = {
                "success": success,
                "retrieved_ticks": len(retrieved_ticks),
                "message": "數據存儲基本功能測試通過" if success else "數據存儲基本功能測試失敗"
            }
            
            self.logger.info(f"測試 {test_name} 結果: {'成功' if success else '失敗'}")
            
            # 清理測試文件
            import shutil
            test_path = Path(test_storage_path)
            if test_path.exists():
                shutil.rmtree(test_path)
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "error": str(e),
                "message": f"數據存儲基本功能測試出錯: {e}"
            }
            handle_error(e, f"測試 {test_name}")
    
    async def test_mt4_data_feed_mock(self):
        """測試 MT4 數據饋送器（模擬模式）"""
        test_name = "mt4_data_feed_mock"
        self.logger.info(f"開始測試: {test_name}")
        
        try:
            # 創建事件循環
            event_loop = EventLoop()
            
            # 創建事件收集器
            received_events = []
            
            async def market_event_handler(event):
                received_events.append(event)
                self.logger.info(f"收到市場事件: {event.symbol}")
            
            event_loop.add_handler(EventType.MARKET, market_event_handler)
            
            # 創建 MT4 數據饋送器
            data_feed = MT4DataFeed(
                symbols=["EURUSD"],
                event_queue=event_loop,
                mt4_bridge=self.mock_bridge,
                timeframes=[TimeFrame.M1],
                enable_tick_collection=True,
                enable_indicators=False
            )
            
            # 啟動事件循環
            event_task = asyncio.create_task(event_loop.run())
            
            # 短暫運行數據饋送器
            feed_task = asyncio.create_task(data_feed.run())
            
            # 等待一段時間
            await asyncio.sleep(2)
            
            # 停止服務
            data_feed.stop()
            event_loop.stop()
            
            # 取消任務
            feed_task.cancel()
            event_task.cancel()
            
            try:
                await asyncio.gather(feed_task, event_task, return_exceptions=True)
            except:
                pass
            
            # 檢查統計信息
            stats = data_feed.get_statistics()
            
            success = stats.get('connected', False)
            
            self.test_results[test_name] = {
                "success": success,
                "connected": stats.get('connected', False),
                "events_received": len(received_events),
                "statistics": stats,
                "message": "MT4 數據饋送器測試通過" if success else "MT4 數據饋送器測試失敗"
            }
            
            self.logger.info(f"測試 {test_name} 結果: {'成功' if success else '失敗'}")
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "error": str(e),
                "message": f"MT4 數據饋送器測試出錯: {e}"
            }
            handle_error(e, f"測試 {test_name}")
    
    def _create_test_dataframe(self):
        """創建測試用的 DataFrame"""
        import pandas as pd
        
        dates = [datetime.now(timezone.utc) - timedelta(minutes=i) for i in range(50, 0, -1)]
        data = []
        
        for i, date in enumerate(dates):
            data.append({
                'Date': date,
                'Open': 1.1000 + i * 0.0001,
                'High': 1.1005 + i * 0.0001,
                'Low': 1.0995 + i * 0.0001,
                'Close': 1.1002 + i * 0.0001,
                'Volume': 1000 + i * 10
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    async def run_all_tests(self):
        """運行所有測試"""
        self.logger.info("開始 MT4 數據收集系統整合測試")
        
        if not IMPORTS_SUCCESS:
            self.logger.error("導入失敗，無法運行測試")
            return
        
        tests = [
            self.test_event_system_compatibility,
            self.test_tick_collector_basic,
            self.test_ohlc_aggregator_basic,
            self.test_data_storage_basic,
            self.test_mt4_data_feed_mock
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                self.logger.error(f"測試執行失敗: {e}")
        
        # 顯示測試結果摘要
        self.print_test_summary()
    
    def print_test_summary(self):
        """打印測試結果摘要"""
        print("\n" + "="*80)
        print("MT4 數據收集系統整合測試結果摘要")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        
        print(f"總測試數: {total_tests}")
        print(f"通過測試: {passed_tests}")
        print(f"失敗測試: {total_tests - passed_tests}")
        print(f"通過率: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ 通過" if result['success'] else "❌ 失敗"
            print(f"{status} {test_name}: {result['message']}")
            
            if not result['success'] and 'error' in result:
                print(f"   錯誤: {result['error']}")
        
        print("="*80)

async def main():
    """主函數"""
    tester = IntegrationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())