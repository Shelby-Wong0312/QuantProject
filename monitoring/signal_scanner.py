"""
信號掃描器 - 多時間框架技術分析
Signal Scanner - Multi-timeframe Technical Analysis
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
import yaml
from pathlib import Path
import sqlite3

# 添加項目根目錄到路徑
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.free_data_client import FreeDataClient

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """信號數據結構"""

    symbol: str
    signal_type: str
    strength: float  # 0-1之間的信號強度
    direction: str  # 'bullish', 'bearish', 'neutral'
    value: float
    timestamp: datetime
    timeframe: str  # '1m', '5m', '1h', '1d'
    metadata: Dict[str, Any] = None


class SignalScanner:
    """
    信號掃描器 - 檢測價格突破、成交量異常、技術指標信號
    """

    def __init__(self, config_path: str = "monitoring/config.yaml"):
        """初始化信號掃描器"""
        self.config = self._load_config(config_path)
        self.client = FreeDataClient()
        self.signals_cache = {}
        self.last_scan_time = {}

        # 信號配置
        self.signal_config = self.config.get("signals", {})
        self.signal_weights = self.config.get("signal_weights", {})

        logger.info("Signal Scanner initialized")

    def _load_config(self, config_path: str) -> Dict:
        """載入配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """預設配置"""
        return {
            "signals": {
                "price_breakout": {
                    "enabled": True,
                    "lookback_periods": 20,
                    "breakout_threshold": 0.02,
                },
                "volume_anomaly": {
                    "enabled": True,
                    "volume_multiplier": 2.5,
                    "lookback_periods": 20,
                },
                "rsi_signals": {
                    "enabled": True,
                    "overbought_threshold": 70,
                    "oversold_threshold": 30,
                    "period": 14,
                },
                "macd_signals": {
                    "enabled": True,
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                },
                "bollinger_bands": {"enabled": True, "period": 20, "std_dev": 2.0},
            },
            "signal_weights": {
                "price_breakout": 0.3,
                "volume_anomaly": 0.25,
                "rsi_signals": 0.15,
                "macd_signals": 0.15,
                "bollinger_bands": 0.15,
            },
        }

    def scan_price_breakout(self, symbol: str, timeframe: str = "1d") -> List[Signal]:
        """
        掃描價格突破信號

        Args:
            symbol: 股票代碼
            timeframe: 時間框架

        Returns:
            突破信號列表
        """
        []

        if not self.signal_config.get("price_breakout", {}).get("enabled", False):
            return signals

        try:
            # 獲取歷史數據
            hist_data = self.client.get_historical_data(
                symbol, period="3mo", interval=timeframe
            )

            if hist_data is None or len(hist_data) < 21:
                return signals

            # 計算指標
            config = self.signal_config["price_breakout"]
            lookback = config.get("lookback_periods", 20)
            threshold = config.get("breakout_threshold", 0.02)

            # 計算阻力和支撐位
            high_resistance = hist_data["High"].rolling(window=lookback).max()
            low_support = hist_data["Low"].rolling(window=lookback).min()

            # 當前價格
            hist_data["Close"].iloc[-1]
            current_high = hist_data["High"].iloc[-1]
            current_low = hist_data["Low"].iloc[-1]
            current_volume = hist_data["Volume"].iloc[-1]

            # 平均成交量
            avg_volume = hist_data["Volume"].rolling(window=lookback).mean().iloc[-1]

            # 向上突破檢測
            resistance_level = high_resistance.iloc[-2]  # 前一期的阻力位
            if (
                current_high > resistance_level
                and (current_high - resistance_level) / resistance_level > threshold
            ):

                # 成交量確認
                volume_confirmed = True
                if config.get("volume_confirmation", True):
                    volume_confirmed = current_volume > avg_volume * 1.5

                if volume_confirmed:
                    strength = min(
                        1.0,
                        (current_high - resistance_level)
                        / resistance_level
                        / threshold,
                    )
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type="price_breakout",
                            strength=strength,
                            direction="bullish",
                            value=current_high,
                            timestamp=datetime.now(),
                            timeframe=timeframe,
                            metadata={
                                "resistance_level": resistance_level,
                                "breakout_magnitude": (current_high - resistance_level)
                                / resistance_level,
                                "volume_ratio": current_volume / avg_volume,
                            },
                        )
                    )

            # 向下突破檢測
            support_level = low_support.iloc[-2]  # 前一期的支撐位
            if (
                current_low < support_level
                and (support_level - current_low) / support_level > threshold
            ):

                # 成交量確認
                volume_confirmed = True
                if config.get("volume_confirmation", True):
                    volume_confirmed = current_volume > avg_volume * 1.5

                if volume_confirmed:
                    strength = min(
                        1.0, (support_level - current_low) / support_level / threshold
                    )
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type="price_breakout",
                            strength=strength,
                            direction="bearish",
                            value=current_low,
                            timestamp=datetime.now(),
                            timeframe=timeframe,
                            metadata={
                                "support_level": support_level,
                                "breakout_magnitude": (support_level - current_low)
                                / support_level,
                                "volume_ratio": current_volume / avg_volume,
                            },
                        )
                    )

        except Exception as e:
            logger.error(f"Error scanning price breakout for {symbol}: {e}")

        return signals

    def scan_volume_anomaly(self, symbol: str, timeframe: str = "1d") -> List[Signal]:
        """
        掃描成交量異常

        Args:
            symbol: 股票代碼
            timeframe: 時間框架

        Returns:
            成交量異常信號列表
        """
        []

        if not self.signal_config.get("volume_anomaly", {}).get("enabled", False):
            return signals

        try:
            # 獲取歷史數據
            hist_data = self.client.get_historical_data(
                symbol, period="2mo", interval=timeframe
            )

            if hist_data is None or len(hist_data) < 21:
                return signals

            config = self.signal_config["volume_anomaly"]
            lookback = config.get("lookback_periods", 20)
            multiplier = config.get("volume_multiplier", 2.5)

            # 計算平均成交量
            avg_volume = hist_data["Volume"].rolling(window=lookback).mean()
            std_volume = hist_data["Volume"].rolling(window=lookback).std()

            current_volume = hist_data["Volume"].iloc[-1]
            avg_vol_current = avg_volume.iloc[-1]
            std_vol_current = std_volume.iloc[-1]

            # 檢測異常成交量
            if current_volume > avg_vol_current * multiplier:
                # 計算Z-score
                z_score = (
                    (current_volume - avg_vol_current) / std_vol_current
                    if std_vol_current > 0
                    else 0
                )
                strength = min(1.0, z_score / 3.0)  # 正規化到0-1

                # 判斷方向（基於價格變化）
                price_change = (
                    hist_data["Close"].iloc[-1] - hist_data["Close"].iloc[-2]
                ) / hist_data["Close"].iloc[-2]
                direction = (
                    "bullish"
                    if price_change > 0
                    else "bearish" if price_change < 0 else "neutral"
                )

                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="volume_anomaly",
                        strength=abs(strength),
                        direction=direction,
                        value=current_volume,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "volume_ratio": current_volume / avg_vol_current,
                            "z_score": z_score,
                            "price_change": price_change,
                            "avg_volume": avg_vol_current,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Error scanning volume anomaly for {symbol}: {e}")

        return signals

    def scan_rsi_signals(self, symbol: str, timeframe: str = "1d") -> List[Signal]:
        """
        掃描RSI信號

        Args:
            symbol: 股票代碼
            timeframe: 時間框架

        Returns:
            RSI信號列表
        """
        []

        if not self.signal_config.get("rsi_signals", {}).get("enabled", False):
            return signals

        try:
            # 獲取歷史數據
            hist_data = self.client.get_historical_data(
                symbol, period="2mo", interval=timeframe
            )

            if hist_data is None or len(hist_data) < 21:
                return signals

            # 計算RSI
            hist_with_indicators = self.client.calculate_indicators(hist_data)

            if "RSI" not in hist_with_indicators.columns:
                return signals

            config = self.signal_config["rsi_signals"]
            overbought = config.get("overbought_threshold", 70)
            oversold = config.get("oversold_threshold", 30)

            current_rsi = hist_with_indicators["RSI"].iloc[-1]
            prev_rsi = hist_with_indicators["RSI"].iloc[-2]

            # 超買信號
            if current_rsi > overbought:
                strength = (current_rsi - overbought) / (100 - overbought)
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="rsi_overbought",
                        strength=strength,
                        direction="bearish",
                        value=current_rsi,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "rsi_value": current_rsi,
                            "threshold": overbought,
                            "previous_rsi": prev_rsi,
                        },
                    )
                )

            # 超賣信號
            elif current_rsi < oversold:
                strength = (oversold - current_rsi) / oversold
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="rsi_oversold",
                        strength=strength,
                        direction="bullish",
                        value=current_rsi,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "rsi_value": current_rsi,
                            "threshold": oversold,
                            "previous_rsi": prev_rsi,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Error scanning RSI signals for {symbol}: {e}")

        return signals

    def scan_macd_signals(self, symbol: str, timeframe: str = "1d") -> List[Signal]:
        """
        掃描MACD信號

        Args:
            symbol: 股票代碼
            timeframe: 時間框架

        Returns:
            MACD信號列表
        """
        []

        if not self.signal_config.get("macd_signals", {}).get("enabled", False):
            return signals

        try:
            # 獲取歷史數據
            hist_data = self.client.get_historical_data(
                symbol, period="3mo", interval=timeframe
            )

            if hist_data is None or len(hist_data) < 50:
                return signals

            # 計算MACD
            hist_with_indicators = self.client.calculate_indicators(hist_data)

            if (
                "MACD" not in hist_with_indicators.columns
                or "MACD_Signal" not in hist_with_indicators.columns
            ):
                return signals

            current_macd = hist_with_indicators["MACD"].iloc[-1]
            current_signal = hist_with_indicators["MACD_Signal"].iloc[-1]
            prev_macd = hist_with_indicators["MACD"].iloc[-2]
            prev_signal = hist_with_indicators["MACD_Signal"].iloc[-2]

            # MACD線上穿信號線（買入信號）
            if current_macd > current_signal and prev_macd <= prev_signal:
                strength = min(
                    1.0,
                    abs(current_macd - current_signal)
                    / max(abs(current_macd), abs(current_signal), 0.001),
                )
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="macd_bullish_crossover",
                        strength=strength,
                        direction="bullish",
                        value=current_macd - current_signal,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "macd": current_macd,
                            "signal": current_signal,
                            "histogram": current_macd - current_signal,
                        },
                    )
                )

            # MACD線下穿信號線（賣出信號）
            elif current_macd < current_signal and prev_macd >= prev_signal:
                strength = min(
                    1.0,
                    abs(current_macd - current_signal)
                    / max(abs(current_macd), abs(current_signal), 0.001),
                )
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="macd_bearish_crossover",
                        strength=strength,
                        direction="bearish",
                        value=current_macd - current_signal,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "macd": current_macd,
                            "signal": current_signal,
                            "histogram": current_macd - current_signal,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Error scanning MACD signals for {symbol}: {e}")

        return signals

    def scan_bollinger_bands(self, symbol: str, timeframe: str = "1d") -> List[Signal]:
        """
        掃描布林通道信號

        Args:
            symbol: 股票代碼
            timeframe: 時間框架

        Returns:
            布林通道信號列表
        """
        []

        if not self.signal_config.get("bollinger_bands", {}).get("enabled", False):
            return signals

        try:
            # 獲取歷史數據
            hist_data = self.client.get_historical_data(
                symbol, period="2mo", interval=timeframe
            )

            if hist_data is None or len(hist_data) < 21:
                return signals

            # 計算布林通道
            hist_with_indicators = self.client.calculate_indicators(hist_data)

            if not all(
                col in hist_with_indicators.columns
                for col in ["BB_Upper", "BB_Lower", "BB_Middle"]
            ):
                return signals

            current_price = hist_with_indicators["Close"].iloc[-1]
            bb_upper = hist_with_indicators["BB_Upper"].iloc[-1]
            bb_lower = hist_with_indicators["BB_Lower"].iloc[-1]
            bb_middle = hist_with_indicators["BB_Middle"].iloc[-1]

            # 價格接觸或突破上軌
            if current_price >= bb_upper:
                strength = min(1.0, (current_price - bb_upper) / (bb_upper - bb_middle))
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="bollinger_upper_touch",
                        strength=abs(strength),
                        direction="bearish",  # 通常認為是超買
                        value=current_price,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "price": current_price,
                            "bb_upper": bb_upper,
                            "bb_middle": bb_middle,
                            "distance_from_upper": (current_price - bb_upper)
                            / bb_upper,
                        },
                    )
                )

            # 價格接觸或突破下軌
            elif current_price <= bb_lower:
                strength = min(1.0, (bb_lower - current_price) / (bb_middle - bb_lower))
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type="bollinger_lower_touch",
                        strength=abs(strength),
                        direction="bullish",  # 通常認為是超賣
                        value=current_price,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        metadata={
                            "price": current_price,
                            "bb_lower": bb_lower,
                            "bb_middle": bb_middle,
                            "distance_from_lower": (bb_lower - current_price)
                            / bb_lower,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Error scanning Bollinger Bands for {symbol}: {e}")

        return signals

    def scan_symbol_comprehensive(
        self, symbol: str, timeframes: List[str] = None
    ) -> List[Signal]:
        """
        對單個股票進行全面信號掃描

        Args:
            symbol: 股票代碼
            timeframes: 時間框架列表

        Returns:
            所有檢測到的信號
        """
        if timeframes is None:
            timeframes = ["1d"]

        all_signals = []

        for timeframe in timeframes:
            try:
                # 掃描各種信號
                []
                signals.extend(self.scan_price_breakout(symbol, timeframe))
                signals.extend(self.scan_volume_anomaly(symbol, timeframe))
                signals.extend(self.scan_rsi_signals(symbol, timeframe))
                signals.extend(self.scan_macd_signals(symbol, timeframe))
                signals.extend(self.scan_bollinger_bands(symbol, timeframe))

                all_signals.extend(signals)

                if signals:
                    logger.debug(
                        f"Found {len(signals)} signals for {symbol} on {timeframe}"
                    )

            except Exception as e:
                logger.error(f"Error scanning {symbol} on {timeframe}: {e}")

        return all_signals

    def calculate_combined_signal_strength(self, signals: List[Signal]) -> float:
        """
        計算組合信號強度

        Args:
            signals: 信號列表

        Returns:
            0-1之間的組合信號強度
        """
        if not signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in signals:
            # 獲取信號權重
            weight = self.signal_weights.get(signal.signal_type.split("_")[0], 0.1)

            # 考慮信號方向的一致性
            direction_multiplier = 1.0
            if signal.direction == "bearish":
                direction_multiplier = -1.0
            elif signal.direction == "neutral":
                direction_multiplier = 0.5

            weighted_sum += signal.strength * weight * direction_multiplier
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # 正規化到0-1範圍
        combined_strength = abs(weighted_sum / total_weight)
        return min(1.0, combined_strength)

    def save_signals_to_db(self, signals: List[Signal]):
        """將信號保存到數據庫"""
        try:
            db_path = self.config.get("system", {}).get(
                "database_path", "data/market_data.db"
            )

            with sqlite3.connect(db_path) as conn:
                # 創建信號表（如果不存在）
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS signal_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        strength REAL NOT NULL,
                        direction TEXT NOT NULL,
                        value REAL NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        metadata TEXT
                    )
                """
                )

                # 插入信號
                for signal in signals:
                    metadata_json = None
                    if signal.metadata:
                        import json

                        metadata_json = json.dumps(signal.metadata)

                    conn.execute(
                        """
                        INSERT INTO signal_history 
                        (symbol, signal_type, strength, direction, value, timeframe, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            signal.symbol,
                            signal.signal_type,
                            signal.strength,
                            signal.direction,
                            signal.value,
                            signal.timeframe,
                            signal.timestamp,
                            metadata_json,
                        ),
                    )

                conn.commit()
                logger.debug(f"Saved {len(signals)} signals to database")

        except Exception as e:
            logger.error(f"Error saving signals to database: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scanner = SignalScanner()

    # 測試單個股票掃描
    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    for symbol in test_symbols:
        print(f"\n掃描 {symbol}:")
        scanner.scan_symbol_comprehensive(symbol)

        if signals:
            for signal in signals:
                print(
                    f"  {signal.signal_type}: {signal.direction} (強度: {signal.strength:.2f})"
                )

            combined_strength = scanner.calculate_combined_signal_strength(signals)
            print(f"  組合信號強度: {combined_strength:.2f}")

            # 保存到數據庫
            scanner.save_signals_to_db(signals)
        else:
            print("  無信號檢測到")

    print("\n信號掃描器測試完成！")
