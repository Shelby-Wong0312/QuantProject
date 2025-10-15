# -*- coding: utf-8 -*-
"""
OHLC 聚合器
將 Tick 數據聚合成不同時間框架的 K 線，並即時計算技術指標
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import threading
import time

from .tick_collector import TickData

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """時間框架枚舉"""

    M1 = "M1"  # 1分鐘
    M5 = "M5"  # 5分鐘
    M15 = "M15"  # 15分鐘
    M30 = "M30"  # 30分鐘
    H1 = "H1"  # 1小時
    H4 = "H4"  # 4小時
    D1 = "D1"  # 1天
    W1 = "W1"  # 1週
    MN1 = "MN1"  # 1月

    @property
    def minutes(self) -> int:
        """獲取時間框架對應的分鐘數"""
        mapping = {
            TimeFrame.M1: 1,
            TimeFrame.M5: 5,
            TimeFrame.M15: 15,
            TimeFrame.M30: 30,
            TimeFrame.H1: 60,
            TimeFrame.H4: 240,
            TimeFrame.D1: 1440,
            TimeFrame.W1: 10080,
            TimeFrame.MN1: 43200,  # 約一個月
        }
        return mapping.get(self, 1)


@dataclass
class OHLCBar:
    """OHLC K線數據結構"""

    symbol: str
    timeframe: TimeFrame
    timestamp: datetime  # K線開始時間
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        data = asdict(self)
        data["timeframe"] = self.timeframe.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCBar":
        """從字典創建 OHLCBar"""
        data["timeframe"] = TimeFrame(data["timeframe"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class TechnicalIndicators:
    """技術指標計算器"""

    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        """簡單移動平均線"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def ema(prices: List[float], period: int, alpha: float = None) -> Optional[float]:
        """指數移動平均線"""
        if len(prices) < period:
            return None

        if alpha is None:
            alpha = 2.0 / (period + 1)

        if len(prices) == period:
            return TechnicalIndicators.sma(prices, period)

        # 遞歸計算 EMA
        prev_ema = TechnicalIndicators.ema(prices[:-1], period, alpha)
        if prev_ema is None:
            return None

        return alpha * prices[-1] + (1 - alpha) * prev_ema

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """相對強弱指標"""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))

    @staticmethod
    def bollinger_bands(
        prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """布林通道 (上軌, 中軌, 下軌)"""
        if len(prices) < period:
            return None, None, None

        sma_val = TechnicalIndicators.sma(prices, period)
        if sma_val is None:
            return None, None, None

        recent_prices = prices[-period:]
        std = np.std(recent_prices)

        upper = sma_val + (std_dev * std)
        middle = sma_val
        lower = sma_val - (std_dev * std)

        return upper, middle, lower

    @staticmethod
    def macd(
        prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """MACD 指標 (MACD線, 信號線, 柱狀圖)"""
        if len(prices) < slow:
            return None, None, None

        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)

        if ema_fast is None or ema_slow is None:
            return None, None, None

        macd_line = ema_fast - ema_slow

        # 計算信號線需要歷史 MACD 值，這裡簡化處理
        return macd_line, None, None


class OHLCAggregator:
    """OHLC 數據聚合器"""

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[TimeFrame] = None,
        enable_indicators: bool = True,
        indicator_periods: Dict[str, int] = None,
        max_bars_per_timeframe: int = 1000,
    ):
        """
        初始化 OHLC 聚合器

        Args:
            symbols: 交易品種列表
            timeframes: 時間框架列表
            enable_indicators: 是否啟用技術指標計算
            indicator_periods: 技術指標週期配置
            max_bars_per_timeframe: 每個時間框架最大保存K線數
        """
        self.symbols = symbols
        self.timeframes = timeframes or [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1]
        self.enable_indicators = enable_indicators
        self.max_bars_per_timeframe = max_bars_per_timeframe

        # 技術指標配置
        self.indicator_periods = indicator_periods or {
            "sma_short": 20,
            "sma_long": 50,
            "ema_short": 12,
            "ema_long": 26,
            "rsi": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }

        # K線數據存儲: {symbol: {timeframe: deque[OHLCBar]}}
        self.ohlc_data: Dict[str, Dict[TimeFrame, deque]] = {}

        # 當前未完成的K線: {symbol: {timeframe: OHLCBar}}
        self.current_bars: Dict[str, Dict[TimeFrame, OHLCBar]] = {}

        # 技術指標數據: {symbol: {timeframe: {indicator: deque}}}
        self.indicators: Dict[str, Dict[TimeFrame, Dict[str, deque]]] = {}

        # 價格歷史用於指標計算: {symbol: {timeframe: deque[float]}}
        self.price_history: Dict[str, Dict[TimeFrame, deque]] = {}

        # 初始化數據結構
        self._initialize_data_structures()

        # 回調函數
        self.bar_callbacks: List[Callable[[OHLCBar], None]] = []
        self.indicator_callbacks: List[Callable[[str, TimeFrame, Dict[str, Any]], None]] = []

        # 線程鎖
        self._lock = threading.RLock()

        logger.info(
            f"OHLC聚合器已初始化，品種: {symbols}, 時間框架: {[tf.value for tf in self.timeframes]}"
        )

    def _initialize_data_structures(self):
        """初始化數據結構"""
        for symbol in self.symbols:
            self.ohlc_data[symbol] = {}
            self.current_bars[symbol] = {}
            self.indicators[symbol] = {}
            self.price_history[symbol] = {}

            for timeframe in self.timeframes:
                self.ohlc_data[symbol][timeframe] = deque(maxlen=self.max_bars_per_timeframe)
                self.current_bars[symbol][timeframe] = None
                self.price_history[symbol][timeframe] = deque(maxlen=200)  # 保存足夠的歷史價格

                if self.enable_indicators:
                    self.indicators[symbol][timeframe] = {
                        "sma_short": deque(maxlen=self.max_bars_per_timeframe),
                        "sma_long": deque(maxlen=self.max_bars_per_timeframe),
                        "ema_short": deque(maxlen=self.max_bars_per_timeframe),
                        "ema_long": deque(maxlen=self.max_bars_per_timeframe),
                        "rsi": deque(maxlen=self.max_bars_per_timeframe),
                        "bb_upper": deque(maxlen=self.max_bars_per_timeframe),
                        "bb_middle": deque(maxlen=self.max_bars_per_timeframe),
                        "bb_lower": deque(maxlen=self.max_bars_per_timeframe),
                        "macd": deque(maxlen=self.max_bars_per_timeframe),
                        "macd_signal": deque(maxlen=self.max_bars_per_timeframe),
                        "macd_histogram": deque(maxlen=self.max_bars_per_timeframe),
                    }

    def add_bar_callback(self, callback: Callable[[OHLCBar], None]):
        """添加K線完成回調"""
        self.bar_callbacks.append(callback)
        logger.info(f"已添加K線回調: {callback.__name__}")

    def add_indicator_callback(self, callback: Callable[[str, TimeFrame, Dict[str, Any]], None]):
        """添加技術指標回調"""
        self.indicator_callbacks.append(callback)
        logger.info(f"已添加指標回調: {callback.__name__}")

    def _get_bar_start_time(self, timestamp: datetime, timeframe: TimeFrame) -> datetime:
        """獲取K線開始時間"""
        if timeframe == TimeFrame.M1:
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == TimeFrame.M5:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.M15:
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.M30:
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.H1:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.H4:
            hour = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.D1:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.W1:
            # 週一開始
            days_since_monday = timestamp.weekday()
            start_of_week = timestamp - timedelta(days=days_since_monday)
            return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.MN1:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp.replace(second=0, microsecond=0)

    def _create_new_bar(self, symbol: str, timeframe: TimeFrame, tick: TickData) -> OHLCBar:
        """創建新的K線"""
        bar_time = self._get_bar_start_time(tick.timestamp, timeframe)

        return OHLCBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=bar_time,
            open=tick.last,
            high=tick.last,
            low=tick.last,
            close=tick.last,
            volume=tick.volume,
            tick_count=1,
        )

    def _update_bar(self, bar: OHLCBar, tick: TickData) -> OHLCBar:
        """更新K線數據"""
        bar.high = max(bar.high, tick.last)
        bar.low = min(bar.low, tick.last)
        bar.close = tick.last
        bar.volume += tick.volume
        bar.tick_count += 1
        return bar

    def _calculate_indicators(self, symbol: str, timeframe: TimeFrame):
        """計算技術指標"""
        if not self.enable_indicators:
            return

        with self._lock:
            prices = list(self.price_history[symbol][timeframe])
            if len(prices) < 2:
                return

            indicators = {}

            try:
                # 移動平均線
                sma_short = TechnicalIndicators.sma(prices, self.indicator_periods["sma_short"])
                sma_long = TechnicalIndicators.sma(prices, self.indicator_periods["sma_long"])
                ema_short = TechnicalIndicators.ema(prices, self.indicator_periods["ema_short"])
                ema_long = TechnicalIndicators.ema(prices, self.indicator_periods["ema_long"])

                # RSI
                rsi = TechnicalIndicators.rsi(prices, self.indicator_periods["rsi"])

                # 布林通道
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                    prices, self.indicator_periods["bb_period"]
                )

                # MACD
                macd, macd_signal, macd_hist = TechnicalIndicators.macd(
                    prices,
                    self.indicator_periods["macd_fast"],
                    self.indicator_periods["macd_slow"],
                    self.indicator_periods["macd_signal"],
                )

                # 存儲指標值
                indicator_data = self.indicators[symbol][timeframe]

                if sma_short is not None:
                    indicator_data["sma_short"].append(sma_short)
                    indicators["sma_short"] = sma_short

                if sma_long is not None:
                    indicator_data["sma_long"].append(sma_long)
                    indicators["sma_long"] = sma_long

                if ema_short is not None:
                    indicator_data["ema_short"].append(ema_short)
                    indicators["ema_short"] = ema_short

                if ema_long is not None:
                    indicator_data["ema_long"].append(ema_long)
                    indicators["ema_long"] = ema_long

                if rsi is not None:
                    indicator_data["rsi"].append(rsi)
                    indicators["rsi"] = rsi

                if bb_upper is not None:
                    indicator_data["bb_upper"].append(bb_upper)
                    indicator_data["bb_middle"].append(bb_middle)
                    indicator_data["bb_lower"].append(bb_lower)
                    indicators["bollinger_bands"] = {
                        "upper": bb_upper,
                        "middle": bb_middle,
                        "lower": bb_lower,
                    }

                if macd is not None:
                    indicator_data["macd"].append(macd)
                    indicators["macd"] = macd

                # 觸發指標回調
                if indicators:
                    for callback in self.indicator_callbacks:
                        try:
                            callback(symbol, timeframe, indicators)
                        except Exception as e:
                            logger.error(f"指標回調函數執行失敗: {e}")

            except Exception as e:
                logger.error(f"計算技術指標時出錯: {e}")

    def process_tick(self, tick: TickData):
        """處理 Tick 數據，聚合成 OHLC"""
        if tick.symbol not in self.symbols:
            return

        with self._lock:
            for timeframe in self.timeframes:
                current_bar = self.current_bars[tick.symbol][timeframe]
                bar_start_time = self._get_bar_start_time(tick.timestamp, timeframe)

                # 檢查是否需要創建新K線
                if current_bar is None or current_bar.timestamp != bar_start_time:
                    # 完成當前K線
                    if current_bar is not None:
                        # 存儲完成的K線
                        self.ohlc_data[tick.symbol][timeframe].append(current_bar)

                        # 更新價格歷史
                        self.price_history[tick.symbol][timeframe].append(current_bar.close)

                        # 計算技術指標
                        self._calculate_indicators(tick.symbol, timeframe)

                        # 觸發K線完成回調
                        for callback in self.bar_callbacks:
                            try:
                                callback(current_bar)
                            except Exception as e:
                                logger.error(f"K線回調函數執行失敗: {e}")

                        logger.debug(
                            f"完成K線: {current_bar.symbol} {current_bar.timeframe.value} "
                            f"OHLC({current_bar.open:.5f}, {current_bar.high:.5f}, "
                            f"{current_bar.low:.5f}, {current_bar.close:.5f})"
                        )

                    # 創建新K線
                    current_bar = self._create_new_bar(tick.symbol, timeframe, tick)
                    self.current_bars[tick.symbol][timeframe] = current_bar
                else:
                    # 更新當前K線
                    current_bar = self._update_bar(current_bar, tick)

    def get_ohlc_dataframe(
        self, symbol: str, timeframe: TimeFrame, count: int = None
    ) -> pd.DataFrame:
        """獲取 OHLC 數據的 DataFrame"""
        if symbol not in self.ohlc_data or timeframe not in self.ohlc_data[symbol]:
            return pd.DataFrame()

        with self._lock:
            bars = list(self.ohlc_data[symbol][timeframe])
            if count:
                bars = bars[-count:]

            if not bars:
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume", "tick_count"]
                )

            data = [bar.to_dict() for bar in bars]
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            return df

    def get_indicators_dataframe(
        self, symbol: str, timeframe: TimeFrame, count: int = None
    ) -> pd.DataFrame:
        """獲取技術指標數據的 DataFrame"""
        if (
            not self.enable_indicators
            or symbol not in self.indicators
            or timeframe not in self.indicators[symbol]
        ):
            return pd.DataFrame()

        with self._lock:
            indicator_data = self.indicators[symbol][timeframe]

            # 獲取對應的時間戳
            bars = list(self.ohlc_data[symbol][timeframe])
            if count:
                bars = bars[-count:]

            if not bars:
                return pd.DataFrame()

            timestamps = [bar.timestamp for bar in bars]

            # 構建指標 DataFrame
            indicator_dict = {"timestamp": timestamps}

            for indicator_name, values in indicator_data.items():
                values_list = list(values)
                if count:
                    values_list = values_list[-count:]

                # 確保長度一致
                if len(values_list) == len(timestamps):
                    indicator_dict[indicator_name] = values_list

            if len(indicator_dict) > 1:  # 除了timestamp還有其他指標
                df = pd.DataFrame(indicator_dict)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                return df

            return pd.DataFrame()

    def get_latest_bar(self, symbol: str, timeframe: TimeFrame) -> Optional[OHLCBar]:
        """獲取最新完成的K線"""
        if (
            symbol not in self.ohlc_data
            or timeframe not in self.ohlc_data[symbol]
            or not self.ohlc_data[symbol][timeframe]
        ):
            return None

        with self._lock:
            return self.ohlc_data[symbol][timeframe][-1]

    def get_current_bar(self, symbol: str, timeframe: TimeFrame) -> Optional[OHLCBar]:
        """獲取當前未完成的K線"""
        if symbol not in self.current_bars or timeframe not in self.current_bars[symbol]:
            return None

        with self._lock:
            return self.current_bars[symbol][timeframe]

    def clear_data(self, symbol: str = None, timeframe: TimeFrame = None):
        """清空數據"""
        with self._lock:
            if symbol is None:
                # 清空所有數據
                self._initialize_data_structures()
                logger.info("已清空所有OHLC數據")
            elif timeframe is None:
                # 清空指定品種的所有時間框架數據
                if symbol in self.ohlc_data:
                    for tf in self.timeframes:
                        self.ohlc_data[symbol][tf].clear()
                        self.current_bars[symbol][tf] = None
                        self.price_history[symbol][tf].clear()
                        if self.enable_indicators:
                            for indicator_name in self.indicators[symbol][tf]:
                                self.indicators[symbol][tf][indicator_name].clear()
                logger.info(f"已清空 {symbol} 的所有數據")
            else:
                # 清空指定品種和時間框架的數據
                if symbol in self.ohlc_data and timeframe in self.ohlc_data[symbol]:
                    self.ohlc_data[symbol][timeframe].clear()
                    self.current_bars[symbol][timeframe] = None
                    self.price_history[symbol][timeframe].clear()
                    if self.enable_indicators:
                        for indicator_name in self.indicators[symbol][timeframe]:
                            self.indicators[symbol][timeframe][indicator_name].clear()
                logger.info(f"已清空 {symbol} {timeframe.value} 的數據")


# === 使用示例 ===


async def example_usage():
    """OHLC 聚合器使用示例"""
    from .tick_collector import TickCollector

    # 設置日誌
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    symbols = ["EURUSD", "GBPUSD"]
    timeframes = [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15]

    # 創建聚合器
    aggregator = OHLCAggregator(symbols=symbols, timeframes=timeframes, enable_indicators=True)

    # 添加回調函數
    def on_bar_complete(bar: OHLCBar):
        print(
            f"K線完成: {bar.symbol} {bar.timeframe.value} "
            f"OHLC({bar.open:.5f}, {bar.high:.5f}, {bar.low:.5f}, {bar.close:.5f})"
        )

    def on_indicators_updated(symbol: str, timeframe: TimeFrame, indicators: Dict[str, Any]):
        print(f"指標更新: {symbol} {timeframe.value} - {indicators}")

    aggregator.add_bar_callback(on_bar_complete)
    aggregator.add_indicator_callback(on_indicators_updated)

    # 創建 Tick 收集器並連接到聚合器
    collector = TickCollector(symbols=symbols)
    collector.add_callback(aggregator.process_tick)

    try:
        # 開始收集和聚合
        await collector.start_collecting()

    except KeyboardInterrupt:
        print("用戶中斷...")
    finally:
        collector.stop_collecting()

        # 顯示聚合結果
        for symbol in symbols:
            for timeframe in timeframes:
                df = aggregator.get_ohlc_dataframe(symbol, timeframe, count=5)
                if not df.empty:
                    print(f"\n=== {symbol} {timeframe.value} 最近5根K線 ===")
                    print(df)


if __name__ == "__main__":
    asyncio.run(example_usage())
