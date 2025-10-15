# -*- coding: utf-8 -*-
"""
MT4數據收集系統 - 整合到主數據管道
負責接收實時Tick數據、高頻K線數據，並提供統一的數據接口
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from pathlib import Path
import sys
from collections import deque
from dataclasses import dataclass, asdict
import threading
import time

# 添加mt4_bridge到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mt4_bridge.connector import MT4Connector, create_default_connector
from mt4_bridge.data_collector import (
    MT4DataCollector as BaseCollector,
    TickData,
    OHLCData,
    TimeFrame,
    DataStorage,
    OHLCAggregator,
    TechnicalIndicators,
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """市場數據結構 - 統一接口"""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    spread: float
    volume: int
    ohlc_1m: Optional[Dict] = None
    ohlc_5m: Optional[Dict] = None
    indicators: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class DataQualityChecker:
    """數據質量檢查器"""

    def __init__(
        self,
        max_spread_ratio: float = 0.01,  # 最大點差比例
        max_price_jump: float = 0.05,  # 最大價格跳動
        min_volume: int = 0,
    ):
        self.max_spread_ratio = max_spread_ratio
        self.max_price_jump = max_price_jump
        self.min_volume = min_volume
        self.last_prices = {}  # {symbol: price}

    def check_tick(self, tick: TickData) -> Tuple[bool, str]:
        """
        檢查Tick數據質量

        Returns:
            (is_valid, error_message)
        """
        # 檢查基本數據完整性
        if not tick.symbol or tick.bid <= 0 or tick.ask <= 0:
            return False, "Invalid price data"

        # 檢查點差
        spread_ratio = (tick.ask - tick.bid) / tick.bid
        if spread_ratio > self.max_spread_ratio:
            return False, f"Spread too wide: {spread_ratio:.4%}"

        # 檢查價格跳動
        mid_price = (tick.bid + tick.ask) / 2
        if tick.symbol in self.last_prices:
            last_price = self.last_prices[tick.symbol]
            price_change = abs(mid_price - last_price) / last_price
            if price_change > self.max_price_jump:
                return False, f"Price jump too large: {price_change:.4%}"

        # 更新最後價格
        self.last_prices[tick.symbol] = mid_price

        # 檢查成交量
        if tick.volume < self.min_volume:
            return False, f"Volume too low: {tick.volume}"

        return True, ""

    def check_ohlc(self, ohlc: OHLCData) -> Tuple[bool, str]:
        """檢查OHLC數據質量"""
        # 檢查OHLC邏輯
        if ohlc.high < ohlc.low:
            return False, "High < Low"
        if ohlc.open < ohlc.low or ohlc.open > ohlc.high:
            return False, "Open outside High-Low range"
        if ohlc.close < ohlc.low or ohlc.close > ohlc.high:
            return False, "Close outside High-Low range"

        # 檢查成交量
        if ohlc.volume < 0:
            return False, "Negative volume"

        return True, ""


class DataCache:
    """數據緩存管理器"""

    def __init__(self, max_tick_cache: int = 10000, max_ohlc_cache: int = 5000):
        self.tick_cache = {}  # {symbol: deque}
        self.ohlc_cache = {}  # {(symbol, timeframe): deque}
        self.max_tick_cache = max_tick_cache
        self.max_ohlc_cache = max_ohlc_cache
        self.lock = threading.Lock()

    def add_tick(self, tick: TickData):
        """添加Tick到緩存"""
        with self.lock:
            if tick.symbol not in self.tick_cache:
                self.tick_cache[tick.symbol] = deque(maxlen=self.max_tick_cache)
            self.tick_cache[tick.symbol].append(tick)

    def add_ohlc(self, ohlc: OHLCData):
        """添加OHLC到緩存"""
        with self.lock:
            key = (ohlc.symbol, ohlc.timeframe)
            if key not in self.ohlc_cache:
                self.ohlc_cache[key] = deque(maxlen=self.max_ohlc_cache)
            self.ohlc_cache[key].append(ohlc)

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[TickData]:
        """獲取最近的Tick數據"""
        with self.lock:
            if symbol in self.tick_cache:
                cache = self.tick_cache[symbol]
                return list(cache)[-count:] if len(cache) > count else list(cache)
            return []

    def get_recent_ohlc(
        self, symbol: str, timeframe: TimeFrame, count: int = 100
    ) -> List[OHLCData]:
        """獲取最近的OHLC數據"""
        with self.lock:
            key = (symbol, timeframe)
            if key in self.ohlc_cache:
                cache = self.ohlc_cache[key]
                return list(cache)[-count:] if len(cache) > count else list(cache)
            return []

    def to_dataframe(self, symbol: str, timeframe: TimeFrame = None) -> pd.DataFrame:
        """轉換為DataFrame格式"""
        if timeframe:
            # 返回OHLC數據
            ohlc_list = self.get_recent_ohlc(symbol, timeframe)
            if ohlc_list:
                [o.to_dict() for o in ohlc_list]
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                return df
        else:
            # 返回Tick數據
            tick_list = self.get_recent_ticks(symbol)
            if tick_list:
                [t.to_dict() for t in tick_list]
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                return df

        return pd.DataFrame()


class MT4DataPipeline:
    """MT4數據管道 - 主數據收集系統"""

    def __init__(
        self,
        connector: MT4Connector = None,
        enable_storage: bool = True,
        enable_cache: bool = True,
        enable_quality_check: bool = True,
    ):
        """
        初始化MT4數據管道

        Args:
            connector: MT4連接器
            enable_storage: 是否啟用持久化存儲
            enable_cache: 是否啟用內存緩存
            enable_quality_check: 是否啟用數據質量檢查
        """
        self.connector = connector or create_default_connector()
        self.base_collector = BaseCollector(self.connector)

        # 初始化組件
        self.storage = DataStorage() if enable_storage else None
        self.cache = DataCache() if enable_cache else None
        self.quality_checker = DataQualityChecker() if enable_quality_check else None
        self.aggregator = OHLCAggregator()
        self.indicators = TechnicalIndicators()

        # 訂閱的品種
        self.subscribed_symbols = set()

        # 回調函數
        self.market_data_callbacks = []

        # 統計信息
        self.stats = {
            "total_ticks": 0,
            "valid_ticks": 0,
            "invalid_ticks": 0,
            "total_bars": 0,
            "errors": 0,
            "start_time": None,
        }

        # 運行狀態
        self._running = False
        self._process_thread = None

    def connect(self) -> bool:
        """連接到MT4"""
        if not self.connector.is_connected():
            success = self.connector.connect()
            if success:
                logger.info("成功連接到MT4")
                return True
            else:
                logger.error("無法連接到MT4")
                return False
        return True

    def disconnect(self):
        """斷開連接"""
        self.stop()
        if self.connector.is_connected():
            self.connector.disconnect()
            logger.info("已斷開MT4連接")

    def start(self):
        """啟動數據收集"""
        if not self.connect():
            return False

        self._running = True
        self.stats["start_time"] = datetime.now()

        # 設置回調
        self.base_collector.add_tick_callback(self._process_tick)
        self.base_collector.add_ohlc_callback(self._process_ohlc)

        # 啟動基礎收集器
        self.base_collector.start()

        # 啟動處理線程
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        logger.info("MT4數據管道已啟動")
        return True

    def stop(self):
        """停止數據收集"""
        self._running = False

        # 停止基礎收集器
        self.base_collector.stop()

        # 等待處理線程結束
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2)

        logger.info("MT4數據管道已停止")

    def subscribe(self, symbols: Union[str, List[str]]):
        """訂閱交易品種"""
        if isinstance(symbols, str):
            [symbols]

        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            self.base_collector.subscribe_symbol(symbol)
            logger.info(f"已訂閱: {symbol}")

    def unsubscribe(self, symbols: Union[str, List[str]]):
        """取消訂閱"""
        if isinstance(symbols, str):
            [symbols]

        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
            self.base_collector.unsubscribe_symbol(symbol)
            logger.info(f"已取消訂閱: {symbol}")

    def _process_tick(self, tick: TickData):
        """處理Tick數據"""
        try:
            self.stats["total_ticks"] += 1

            # 質量檢查
            if self.quality_checker:
                is_valid, error_msg = self.quality_checker.check_tick(tick)
                if not is_valid:
                    self.stats["invalid_ticks"] += 1
                    logger.warning(f"Tick數據質量問題 {tick.symbol}: {error_msg}")
                    return

            self.stats["valid_ticks"] += 1

            # 添加到緩存
            if self.cache:
                self.cache.add_tick(tick)

            # 保存到存儲
            if self.storage:
                self.storage.save_tick(tick)

            # 聚合K線
            completed_bars = self.aggregator.add_tick(tick)
            for bar in completed_bars:
                self._process_ohlc(bar)

            # 創建市場數據
            market_data = self._create_market_data(tick)

            # 觸發回調
            self._trigger_callbacks(market_data)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"處理Tick數據錯誤: {e}")

    def _process_ohlc(self, ohlc: OHLCData):
        """處理OHLC數據"""
        try:
            self.stats["total_bars"] += 1

            # 質量檢查
            if self.quality_checker:
                is_valid, error_msg = self.quality_checker.check_ohlc(ohlc)
                if not is_valid:
                    logger.warning(f"OHLC數據質量問題 {ohlc.symbol}: {error_msg}")
                    return

            # 添加到緩存
            if self.cache:
                self.cache.add_ohlc(ohlc)

            # 保存到存儲
            if self.storage:
                self.storage.save_ohlc(ohlc)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"處理OHLC數據錯誤: {e}")

    def _create_market_data(self, tick: TickData) -> MarketData:
        """創建統一的市場數據"""
        mid_price = (tick.bid + tick.ask) / 2

        # 獲取最新K線數據
        ohlc_1m = None
        ohlc_5m = None

        if self.cache:
            bars_1m = self.cache.get_recent_ohlc(tick.symbol, TimeFrame.M1, 1)
            if bars_1m:
                bar = bars_1m[-1]
                ohlc_1m = {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }

            bars_5m = self.cache.get_recent_ohlc(tick.symbol, TimeFrame.M5, 1)
            if bars_5m:
                bar = bars_5m[-1]
                ohlc_5m = {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }

        # 計算基本指標
        indicators = self._calculate_realtime_indicators(tick.symbol)

        return MarketData(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            bid=tick.bid,
            ask=tick.ask,
            mid=mid_price,
            spread=tick.spread,
            volume=tick.volume,
            ohlc_1m=ohlc_1m,
            ohlc_5m=ohlc_5m,
            indicators=indicators,
        )

    def _calculate_realtime_indicators(self, symbol: str) -> Dict:
        """計算實時技術指標"""
        indicators = {}

        if not self.cache:
            return indicators

        # 獲取最近的K線數據
        bars = self.cache.get_recent_ohlc(symbol, TimeFrame.M5, 100)
        if len(bars) < 20:
            return indicators

        # 提取收盤價
        closes = [bar.close for bar in bars]

        # 計算SMA
        if len(closes) >= 20:
            sma20 = sum(closes[-20:]) / 20
            indicators["sma20"] = sma20

        if len(closes) >= 50:
            sma50 = sum(closes[-50:]) / 50
            indicators["sma50"] = sma50

        # 計算RSI
        if len(closes) >= 15:
            rsi_values = self.indicators.rsi(closes, 14)
            if rsi_values:
                indicators["rsi14"] = rsi_values[-1]

        # 計算布林帶
        if len(closes) >= 20:
            bb = self.indicators.bollinger_bands(closes, 20, 2.0)
            if bb["upper"]:
                indicators["bb_upper"] = bb["upper"][-1]
                indicators["bb_middle"] = bb["middle"][-1]
                indicators["bb_lower"] = bb["lower"][-1]

        return indicators

    def _trigger_callbacks(self, market_data: MarketData):
        """觸發市場數據回調"""
        for callback in self.market_data_callbacks:
            try:
                callback(market_data)
            except Exception as e:
                logger.error(f"回調函數錯誤: {e}")

    def _process_loop(self):
        """處理循環 - 定期任務"""
        while self._running:
            try:
                # 定期保存統計信息
                if (
                    self.stats["total_ticks"] % 1000 == 0
                    and self.stats["total_ticks"] > 0
                ):
                    self._save_stats()

                # 定期清理過期數據
                if (
                    self.stats["total_ticks"] % 10000 == 0
                    and self.stats["total_ticks"] > 0
                ):
                    self._cleanup_old_data()

                time.sleep(1)

            except Exception as e:
                logger.error(f"處理循環錯誤: {e}")

    def _save_stats(self):
        """保存統計信息"""
        stats_file = project_root / "logs" / "mt4_pipeline_stats.json"
        stats_file.parent.mkdir(exist_ok=True)

        with open(stats_file, "w") as f:
            json.dump(self.get_stats(), f, indent=2, default=str)

    def _cleanup_old_data(self):
        """清理過期數據"""
        # 這裡可以實現數據清理邏輯
        pass

    def add_callback(self, callback):
        """添加市場數據回調函數"""
        self.market_data_callbacks.append(callback)

    def remove_callback(self, callback):
        """移除回調函數"""
        if callback in self.market_data_callbacks:
            self.market_data_callbacks.remove(callback)

    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """獲取最新的市場數據"""
        if not self.cache:
            return None

        ticks = self.cache.get_recent_ticks(symbol, 1)
        if ticks:
            return self._create_market_data(ticks[0])
        return None

    def get_dataframe(
        self, symbol: str, timeframe: TimeFrame = None, periods: int = 100
    ) -> pd.DataFrame:
        """
        獲取DataFrame格式的數據

        Args:
            symbol: 交易品種
            timeframe: 時間框架，None表示Tick數據
            periods: 數據條數

        Returns:
            pd.DataFrame: 數據框架
        """
        if self.cache:
            df = self.cache.to_dataframe(symbol, timeframe)
            if not df.empty and len(df) > periods:
                return df.tail(periods)
            return df
        return pd.DataFrame()

    def get_indicators(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.M5,
        indicators: List[str] = None,
    ) -> Dict:
        """
        獲取技術指標

        Args:
            symbol: 交易品種
            timeframe: 時間框架
            indicators: 指標列表，None表示所有指標

        Returns:
            Dict: 指標值字典
        """
        if not indicators:
            indicators = [
                "sma20",
                "sma50",
                "rsi14",
                "bb_upper",
                "bb_middle",
                "bb_lower",
            ]

        result = {}

        # 獲取數據
        if self.cache:
            bars = self.cache.get_recent_ohlc(symbol, timeframe, 100)
            if len(bars) < 20:
                return result

            closes = [bar.close for bar in bars]

            # 計算指標
            for ind in indicators:
                if ind == "sma20" and len(closes) >= 20:
                    result[ind] = sum(closes[-20:]) / 20
                elif ind == "sma50" and len(closes) >= 50:
                    result[ind] = sum(closes[-50:]) / 50
                elif ind == "rsi14" and len(closes) >= 15:
                    rsi_values = self.indicators.rsi(closes, 14)
                    if rsi_values:
                        result[ind] = rsi_values[-1]
                elif ind.startswith("bb_") and len(closes) >= 20:
                    bb = self.indicators.bollinger_bands(closes, 20, 2.0)
                    if ind == "bb_upper" and bb["upper"]:
                        result[ind] = bb["upper"][-1]
                    elif ind == "bb_middle" and bb["middle"]:
                        result[ind] = bb["middle"][-1]
                    elif ind == "bb_lower" and bb["lower"]:
                        result[ind] = bb["lower"][-1]

        return result

    def get_stats(self) -> Dict:
        """獲取統計信息"""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            "running": self._running,
            "connected": self.connector.is_connected() if self.connector else False,
            "subscribed_symbols": list(self.subscribed_symbols),
            "uptime_seconds": uptime,
            "total_ticks": self.stats["total_ticks"],
            "valid_ticks": self.stats["valid_ticks"],
            "invalid_ticks": self.stats["invalid_ticks"],
            "total_bars": self.stats["total_bars"],
            "errors": self.stats["errors"],
            "ticks_per_second": self.stats["total_ticks"] / uptime if uptime else 0,
            "validity_rate": (
                self.stats["valid_ticks"] / self.stats["total_ticks"]
                if self.stats["total_ticks"] > 0
                else 0
            ),
        }


# 全局實例
_global_pipeline = None


def get_pipeline() -> MT4DataPipeline:
    """獲取全局數據管道實例"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = MT4DataPipeline()
    return _global_pipeline


def create_pipeline(**kwargs) -> MT4DataPipeline:
    """創建新的數據管道實例"""
    return MT4DataPipeline(**kwargs)


# 便利函數
def start_data_collection(symbols: List[str] = None) -> MT4DataPipeline:
    """快速啟動數據收集"""
    pipeline = get_pipeline()

    if pipeline.start():
        if symbols:
            pipeline.subscribe(symbols)
        logger.info("數據收集已啟動")
        return pipeline
    else:
        logger.error("數據收集啟動失敗")
        return None


def stop_data_collection():
    """停止數據收集"""
    pipeline = get_pipeline()
    pipeline.stop()
    logger.info("數據收集已停止")


def get_realtime_data(symbol: str) -> Optional[MarketData]:
    """獲取實時市場數據"""
    pipeline = get_pipeline()
    return pipeline.get_latest_data(symbol)


def get_historical_data(
    symbol: str, timeframe: str = "M5", periods: int = 100
) -> pd.DataFrame:
    """獲取歷史數據"""
    pipeline = get_pipeline()
    tf = TimeFrame(timeframe)
    return pipeline.get_dataframe(symbol, tf, periods)


if __name__ == "__main__":
    # 測試代碼
    logging.basicConfig(level=logging.DEBUG)

    # 創建數據管道
    pipeline = create_pipeline()

    # 定義回調函數
    def on_market_data(data: MarketData):
        print(f"收到數據: {data.symbol} @ {data.timestamp}")
        print(f"  Bid: {data.bid}, Ask: {data.ask}, Spread: {data.spread}")
        if data.indicators:
            print(f"  指標: {data.indicators}")

    # 添加回調
    pipeline.add_callback(on_market_data)

    # 啟動收集
    if pipeline.start():
        # 訂閱品種
        pipeline.subscribe(["EURUSD", "GBPUSD"])

        print("數據收集已啟動，按Ctrl+C停止...")

        try:
            # 運行10秒
            time.sleep(10)

            # 顯示統計
            stats = pipeline.get_stats()
            print("\n統計信息:")
            print(f"  總Tick數: {stats['total_ticks']}")
            print(f"  有效率: {stats['validity_rate']:.2%}")
            print(f"  每秒Tick: {stats['ticks_per_second']:.2f}")

        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            pipeline.stop()
            print("已停止")
