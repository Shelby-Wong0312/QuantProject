# -*- coding: utf-8 -*-
"""
MT4 數據饋送器
整合到現有的事件驅動架構，與 live_feed.py 提供相同的接口
"""

import asyncio
import logging
import pandas as pd
from collections import deque
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# 導入核心事件系統
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.event import MarketEvent
from core.event_loop import EventLoop
from .tick_collector import TickCollector, TickData
from .ohlc_aggregator import OHLCAggregator, TimeFrame, OHLCBar
from ..zeromq.python_side import MT4Bridge

logger = logging.getLogger(__name__)


class MT4DataFeed:
    """
    MT4 數據饋送器
    與現有的 LiveDataFeed 提供相同的接口，但使用 MT4 作為數據源
    """

    def __init__(
        self,
        symbols: List[str],
        event_queue: EventLoop,
        mt4_bridge: Optional[MT4Bridge] = None,
        timeframes: List[TimeFrame] = None,
        enable_tick_collection: bool = True,
        enable_indicators: bool = True,
        market_event_timeframe: TimeFrame = TimeFrame.M1,
        price_history_length: int = 200,
    ):
        """
        初始化 MT4 數據饋送器

        Args:
            symbols: 交易品種列表
            event_queue: 事件隊列
            mt4_bridge: MT4 橋接實例
            timeframes: 要聚合的時間框架
            enable_tick_collection: 是否啟用 Tick 收集
            enable_indicators: 是否啟用技術指標
            market_event_timeframe: 發送 MarketEvent 的時間框架
            price_history_length: 價格歷史長度
        """
        self.symbols = symbols
        self.event_queue = event_queue
        self.mt4_bridge = mt4_bridge or MT4Bridge()
        self.timeframes = timeframes or [
            TimeFrame.M1,
            TimeFrame.M5,
            TimeFrame.M15,
            TimeFrame.H1,
        ]
        self.enable_tick_collection = enable_tick_collection
        self.enable_indicators = enable_indicators
        self.market_event_timeframe = market_event_timeframe
        self.price_history_length = price_history_length

        # 控制標誌
        self._running = False
        self._connected = False

        # 價格歷史數據（為了與現有系統兼容）
        self.price_history = {
            symbol: deque(maxlen=price_history_length) for symbol in self.symbols
        }

        # 線程池
        self.executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="MT4DataFeed"
        )

        # 初始化數據收集組件
        if self.enable_tick_collection:
            self._initialize_tick_collector()

        self._initialize_ohlc_aggregator()

        logger.info(f"MT4數據饋送器已初始化，監控品種: {symbols}")

    def _initialize_tick_collector(self):
        """初始化 Tick 收集器"""
        try:
            self.tick_collector = TickCollector(
                self.symbols,
                mt4_bridge=self.mt4_bridge,
                cache_size=5000,
                storage_path="./data/mt4_ticks",
                auto_save_interval=300,
            )

            # 添加 Tick 處理回調
            self.tick_collector.add_callback(self._on_tick_received)

            logger.info("Tick收集器已初始化")

        except Exception as e:
            logger.error(f"初始化Tick收集器失敗: {e}")
            self.tick_collector = None

    def _initialize_ohlc_aggregator(self):
        """初始化 OHLC 聚合器"""
        try:
            self.ohlc_aggregator = OHLCAggregator(
                self.symbols,
                timeframes=self.timeframes,
                enable_indicators=self.enable_indicators,
            )

            # 添加K線完成回調
            self.ohlc_aggregator.add_bar_callback(self._on_bar_completed)

            # 如果啟用了指標，添加指標回調
            if self.enable_indicators:
                self.ohlc_aggregator.add_indicator_callback(self._on_indicators_updated)

            logger.info("OHLC聚合器已初始化")

        except Exception as e:
            logger.error(f"初始化OHLC聚合器失敗: {e}")
            self.ohlc_aggregator = None

    def _on_tick_received(self, tick: TickData):
        """Tick 數據接收回調"""
        try:
            # 將 Tick 數據傳遞給 OHLC 聚合器
            if self.ohlc_aggregator:
                self.ohlc_aggregator.process_tick(tick)

            logger.debug(f"處理Tick: {tick.symbol} {tick.last:.5f} @{tick.timestamp}")

        except Exception as e:
            logger.error(f"處理Tick數據時出錯: {e}")

    def _on_bar_completed(self, bar: OHLCBar):
        """K線完成回調"""
        try:
            # 更新價格歷史（為了與現有系統兼容）
            if bar.symbol in self.price_history:
                price_data = {
                    "Date": bar.timestamp,
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "Volume": bar.volume,
                }
                self.price_history[bar.symbol].append(price_data)

            # 如果是指定的時間框架，發送 MarketEvent
            if bar.timeframe == self.market_event_timeframe:
                asyncio.create_task(self._send_market_event(bar))

            logger.debug(
                f"K線完成: {bar.symbol} {bar.timeframe.value} "
                f"OHLC({bar.open:.5f}, {bar.high:.5f}, {bar.low:.5f}, {bar.close:.5f})"
            )

        except Exception as e:
            logger.error(f"處理K線完成事件時出錯: {e}")

    def _on_indicators_updated(
        self, symbol: str, timeframe: TimeFrame, indicators: Dict[str, Any]
    ):
        """技術指標更新回調"""
        try:
            logger.debug(
                f"指標更新: {symbol} {timeframe.value} - 已更新 {len(indicators)} 個指標"
            )

            # 可以在這裡添加指標相關的事件處理

        except Exception as e:
            logger.error(f"處理指標更新時出錯: {e}")

    async def _send_market_event(self, bar: OHLCBar):
        """發送市場事件（與現有系統兼容）"""
        try:
            # 構建與現有 LiveDataFeed 相同格式的 DataFrame
            if (
                bar.symbol in self.price_history
                and len(self.price_history[bar.symbol]) >= 50
            ):
                df_data = list(self.price_history[bar.symbol])[-50:]  # 取最近50條
                df = pd.DataFrame(df_data).set_index("Date")

                # 創建 MarketEvent
                market_event = MarketEvent(
                    symbol=bar.symbol, timestamp=bar.timestamp, ohlcv_data=df
                )

                # 發送事件
                await self.event_queue.put_event(market_event)
                logger.debug(f"已發送 {bar.symbol} 的市場數據事件")

        except Exception as e:
            logger.error(f"發送市場事件時出錯: {e}")

    async def _connect_to_mt4(self) -> bool:
        """連接到 MT4"""
        try:
            # 測試 MT4 連接
            account_info = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.mt4_bridge.get_account_info
            )

            if account_info and "balance" in account_info:
                self._connected = True
                logger.info(f"✅ 已連接到 MT4，帳戶餘額: {account_info.get('balance')}")
                return True
            else:
                logger.error("❌ 無法獲取 MT4 帳戶信息")
                return False

        except Exception as e:
            logger.error(f"❌ 連接 MT4 時出錯: {e}")
            return False

    async def run(self):
        """
        運行數據饋送器（與 LiveDataFeed.run() 相同的接口）
        """
        logger.info("MT4數據饋送器服務已啟動...")

        # 連接到 MT4
        if not await self._connect_to_mt4():
            logger.error("無法連接到 MT4，數據饋送器服務無法啟動。")
            return

        self._running = True

        try:
            # 啟動 Tick 收集器
            if self.tick_collector:
                collect_task = asyncio.create_task(
                    self.tick_collector.start_collecting()
                )

            # 主循環 - 監控連接狀態和處理其他任務
            while self._running:
                try:
                    # 定期檢查 MT4 連接狀態
                    if not await self._check_mt4_connection():
                        logger.warning("MT4 連接丟失，嘗試重新連接...")
                        if not await self._connect_to_mt4():
                            logger.error("重新連接失敗，等待下次嘗試...")
                            await asyncio.sleep(10)
                            continue

                    # 處理其他定期任務
                    await self._periodic_tasks()

                    await asyncio.sleep(5)  # 每5秒檢查一次

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"數據饋送器主循環出錯: {e}")
                    await asyncio.sleep(1)

        finally:
            logger.info("正在停止 MT4數據饋送器...")

            # 停止 Tick 收集器
            if self.tick_collector:
                self.tick_collector.stop_collecting()
                if "collect_task" in locals():
                    collect_task.cancel()
                    try:
                        await collect_task
                    except asyncio.CancelledError:
                        pass

            logger.info("MT4數據饋送器服務已停止。")

    async def _check_mt4_connection(self) -> bool:
        """檢查 MT4 連接狀態"""
        try:
            account_info = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.mt4_bridge.get_account_info()
            )
            return account_info is not None and "balance" in account_info

        except Exception as e:
            logger.debug(f"檢查 MT4 連接時出錯: {e}")
            return False

    async def _periodic_tasks(self):
        """定期任務"""
        try:
            # 可以在這裡添加定期執行的任務
            # 例如：數據清理、統計更新等
            pass

        except Exception as e:
            logger.error(f"執行定期任務時出錯: {e}")

    def stop(self):
        """
        停止數據饋送器（與 LiveDataFeed.stop() 相同的接口）
        """
        logger.info("正在停止 MT4數據饋送器...")
        self._running = False

        # 停止 Tick 收集器
        if self.tick_collector:
            self.tick_collector.stop_collecting()

    # === 與現有系統兼容的方法 ===

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """獲取最新價格（兼容現有系統）"""
        try:
            if self.tick_collector:
                latest_tick = self.tick_collector.get_latest_tick(symbol)
                if latest_tick:
                    return latest_tick.last

            # 如果沒有 Tick 數據，從 OHLC 獲取
            if self.ohlc_aggregator:
                latest_bar = self.ohlc_aggregator.get_current_bar(
                    symbol, self.market_event_timeframe
                )
                if latest_bar:
                    return latest_bar.close

            return None

        except Exception as e:
            logger.error(f"獲取最新價格時出錯: {e}")
            return None

    def get_price_history_df(self, symbol: str, count: int = None) -> pd.DataFrame:
        """獲取價格歷史 DataFrame（兼容現有系統）"""
        try:
            if symbol in self.price_history:
                history = list(self.price_history[symbol])
                if count:
                    history = history[-count:]

                if history:
                    df = pd.DataFrame(history)
                    df.set_index("Date", inplace=True)
                    return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"獲取價格歷史時出錯: {e}")
            return pd.DataFrame()

    def get_ohlc_data(
        self, symbol: str, timeframe: TimeFrame, count: int = None
    ) -> pd.DataFrame:
        """獲取 OHLC 數據"""
        if self.ohlc_aggregator:
            return self.ohlc_aggregator.get_ohlc_dataframe(symbol, timeframe, count)
        return pd.DataFrame()

    def get_indicators_data(
        self, symbol: str, timeframe: TimeFrame, count: int = None
    ) -> pd.DataFrame:
        """獲取技術指標數據"""
        if self.ohlc_aggregator and self.enable_indicators:
            return self.ohlc_aggregator.get_indicators_dataframe(
                symbol, timeframe, count
            )
        return pd.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息"""
        stats = {
            "connected": self._connected,
            "running": self._running,
            "symbols": self.symbols,
            "timeframes": [tf.value for tf in self.timeframes],
            "tick_collection_enabled": self.enable_tick_collection,
            "indicators_enabled": self.enable_indicators,
        }

        # 添加 Tick 收集器統計
        if self.tick_collector:
            tick_stats = self.tick_collector.get_statistics()
            stats.update(
                {
                    "total_ticks": tick_stats.get("total_ticks", 0),
                    "ticks_per_symbol": tick_stats.get("ticks_per_symbol", {}),
                    "average_ticks_per_second": tick_stats.get(
                        "average_ticks_per_second", 0
                    ),
                }
            )

        return stats


# === 使用示例 ===


async def example_usage():
    """MT4 數據饋送器使用示例"""
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    from core.event_loop import EventLoop
    from core.event import EventType

    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 創建事件循環
    event_loop = EventLoop()

    # 市場事件處理器
    async def handle_market_event(event):
        print(f"收到市場事件: {event.symbol} @{event.timestamp}")
        print(f"數據形狀: {event.ohlcv_data.shape}")
        print(f"最新價格: {event.ohlcv_data['Close'].iloc[-1]:.5f}")

    # 註冊事件處理器
    event_loop.add_handler(EventType.MARKET, handle_market_event)

    # 創建 MT4 數據饋送器
    ["EURUSD", "GBPUSD"]
    data_feed = MT4DataFeed(
        symbols,
        event_queue=event_loop,
        timeframes=[TimeFrame.M1, TimeFrame.M5, TimeFrame.M15],
        enable_tick_collection=True,
        enable_indicators=True,
    )

    # 啟動事件循環和數據饋送器
    try:
        # 創建任務
        event_loop_task = asyncio.create_task(event_loop.run())
        data_feed_task = asyncio.create_task(data_feed.run())

        # 等待任務完成
        await asyncio.gather(event_loop_task, data_feed_task)

    except KeyboardInterrupt:
        print("用戶中斷...")
    finally:
        # 清理
        data_feed.stop()
        event_loop.stop()

        # 顯示統計信息
        stats = data_feed.get_statistics()
        print("\n=== 數據饋送器統計 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())
