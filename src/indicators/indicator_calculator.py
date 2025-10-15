"""
Indicator Calculator - Batch calculation engine for technical indicators
技術指標批量計算引擎 - 支援多時間框架和向量化運算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
from pathlib import Path
import pickle
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import all indicator classes
from .trend_indicators import SMA, EMA, WMA, VWAP, MovingAverageCrossover
from .momentum_indicators import RSI, MACD, Stochastic, WilliamsR, CCI
from .volatility_indicators import BollingerBands, ATR, KeltnerChannel, DonchianChannel
from .volume_indicators import OBV, VolumeSMA, MFI, ADLine

logger = logging.getLogger(__name__)


@dataclass
class CalculationConfig:
    """計算配置"""

    timeframes: List[str] = None  # ['1m', '5m', '15m', '1h', '1d']
    indicators: List[str] = None  # 要計算的指標列表
    batch_size: int = 100  # 批量處理大小
    use_multiprocessing: bool = True  # 是否使用多進程
    max_workers: int = None  # 最大工作進程數
    cache_results: bool = True  # 是否緩存結果
    cache_dir: str = "cache/indicators"  # 緩存目錄

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h", "1d"]
        if self.indicators is None:
            self.indicators = [
                "SMA_20",
                "EMA_20",
                "WMA_20",
                "VWAP",
                "RSI_14",
                "MACD",
                "Stochastic",
                "WilliamsR_14",
                "CCI_20",
                "BollingerBands",
                "ATR_14",
                "KeltnerChannel",
                "DonchianChannel",
                "OBV",
                "VolumeSMA_20",
                "MFI_14",
                "ADLine",
            ]
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)


class IndicatorCalculator:
    """
    高效能技術指標批量計算器

    特點：
    1. 向量化計算 - 使用 pandas/numpy 加速
    2. 多進程並行 - 支援大量股票批量處理
    3. 智能緩存 - 避免重複計算
    4. 多時間框架 - 同時生成不同週期指標
    5. 記憶體優化 - 分批處理避免記憶體溢出
    """

    def __init__(self, config: Optional[CalculationConfig] = None):
        """
        初始化計算器

        Args:
            config: 計算配置
        """
        self.config = config or CalculationConfig()

        # 創建緩存目錄
        if self.config.cache_results:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # 初始化指標實例
        self.indicators = self._initialize_indicators()

        # 性能統計
        self.stats = {
            "total_calculations": 0,
            "cache_hits": 0,
            "total_time": 0,
            "stocks_processed": 0,
        }

        logger.info(f"IndicatorCalculator initialized with {len(self.indicators)} indicators")
        logger.info(f"Timeframes: {self.config.timeframes}")
        logger.info(f"Max workers: {self.config.max_workers}")

    def _initialize_indicators(self) -> Dict[str, Any]:
        """初始化所有指標實例"""
        indicators = {}

        # 趨勢指標
        indicators["SMA_20"] = SMA(period=20)
        indicators["SMA_50"] = SMA(period=50)
        indicators["EMA_20"] = EMA(period=20)
        indicators["EMA_50"] = EMA(period=50)
        indicators["WMA_20"] = WMA(period=20)
        indicators["VWAP"] = VWAP()
        indicators["GoldenCross"] = MovingAverageCrossover(fast_period=50, slow_period=200)

        # 動量指標
        indicators["RSI_14"] = RSI(period=14)
        indicators["MACD"] = MACD(fast_period=12, slow_period=26, signal_period=9)
        indicators["Stochastic"] = Stochastic(k_period=14, d_period=3)
        indicators["WilliamsR_14"] = WilliamsR(period=14)
        indicators["CCI_20"] = CCI(period=20)

        # 波動率指標
        indicators["BollingerBands"] = BollingerBands(period=20, std_dev=2.0)
        indicators["ATR_14"] = ATR(period=14)
        indicators["KeltnerChannel"] = KeltnerChannel(ema_period=20, atr_period=10)
        indicators["DonchianChannel"] = DonchianChannel(period=20)

        # 成交量指標
        indicators["OBV"] = OBV()
        indicators["VolumeSMA_20"] = VolumeSMA(period=20)
        indicators["MFI_14"] = MFI(period=14)
        indicators["ADLine"] = ADLine()

        return indicators

    def calculate_all_indicators(
        self, stocks_data: Dict[str, pd.DataFrame], timeframes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        批量計算所有股票的所有指標

        Args:
            stocks_data: {symbol: DataFrame} 股票數據字典
            timeframes: 時間框架列表，如果為None則使用配置中的

        Returns:
            {symbol: {timeframe: {indicator: result}}} 三層嵌套字典
        """
        start_time = time.time()
        timeframes = timeframes or self.config.timeframes

        logger.info(
            f"Starting calculation for {len(stocks_data)} stocks across {len(timeframes)} timeframes"
        )

        # 分批處理股票數據
        stock_symbols = list(stocks_data.keys())
        batches = [
            stock_symbols[i : i + self.config.batch_size]
            for i in range(0, len(stock_symbols), self.config.batch_size)
        ]

        results = {}

        if self.config.use_multiprocessing and len(batches) > 1:
            # 多進程處理
            results = self._calculate_multiprocess(stocks_data, timeframes, batches)
        else:
            # 單進程處理
            for batch in batches:
                batch_data = {symbol: stocks_data[symbol] for symbol in batch}
                batch_results = self._calculate_batch(batch_data, timeframes)
                results.update(batch_results)

        # 更新統計信息
        self.stats["total_time"] += time.time() - start_time
        self.stats["stocks_processed"] += len(stocks_data)

        logger.info(f"Calculation completed in {time.time() - start_time:.2f}s")
        logger.info(
            f"Cache hit rate: {self.stats['cache_hits']/max(1, self.stats['total_calculations'])*100:.1f}%"
        )

        return results

    def _calculate_multiprocess(
        self, stocks_data: Dict[str, pd.DataFrame], timeframes: List[str], batches: List[List[str]]
    ) -> Dict:
        """多進程批量計算"""
        logger.info(f"Using multiprocessing with {self.config.max_workers} workers")

        # 準備參數
        calc_func = partial(
            self._calculate_batch_worker,
            stocks_data=stocks_data,
            timeframes=timeframes,
            config=self.config,
        )

        results = {}

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交所有批次
            futures = {executor.submit(calc_func, batch): batch for batch in batches}

            # 收集結果
            for future in futures:
                try:
                    batch_results = future.result(timeout=300)  # 5分鐘超時
                    results.update(batch_results)
                except Exception as e:
                    batch = futures[future]
                    logger.error(f"Error processing batch {batch}: {e}")

        return results

    @staticmethod
    def _calculate_batch_worker(
        batch: List[str],
        stocks_data: Dict[str, pd.DataFrame],
        timeframes: List[str],
        config: CalculationConfig,
    ) -> Dict:
        """工作進程函數"""
        # 重新初始化計算器（避免序列化問題）
        calculator = IndicatorCalculator(config)

        # 提取批次數據
        batch_data = {symbol: stocks_data[symbol] for symbol in batch}

        return calculator._calculate_batch(batch_data, timeframes)

    def _calculate_batch(self, batch_data: Dict[str, pd.DataFrame], timeframes: List[str]) -> Dict:
        """計算單個批次"""
        results = {}

        for symbol in batch_data:
            try:
                results[symbol] = self._calculate_single_stock(
                    symbol, batch_data[symbol], timeframes
                )
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
                results[symbol] = {}

        return results

    def _calculate_single_stock(
        self, symbol: str, data: pd.DataFrame, timeframes: List[str]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """計算單隻股票的所有指標"""
        results = {}

        for timeframe in timeframes:
            # 重採樣到指定時間框架
            resampled_data = self._resample_data(data, timeframe)

            if resampled_data is None or len(resampled_data) < 50:
                logger.warning(f"Insufficient data for {symbol} at {timeframe}")
                continue

            # 檢查緩存
            cache_key = self._get_cache_key(symbol, timeframe, resampled_data)
            cached_result = self._load_from_cache(cache_key)

            if cached_result is not None:
                results[timeframe] = cached_result
                self.stats["cache_hits"] += 1
                continue

            # 計算所有指標
            timeframe_results = {}

            for indicator_name in self.config.indicators:
                if indicator_name not in self.indicators:
                    continue

                try:
                    indicator = self.indicators[indicator_name]
                    result = indicator.calculate(resampled_data)

                    # 處理不同類型的返回值
                    if isinstance(result, pd.DataFrame):
                        timeframe_results[indicator_name] = result
                    elif isinstance(result, pd.Series):
                        timeframe_results[indicator_name] = result.to_frame(indicator_name)

                    self.stats["total_calculations"] += 1

                except Exception as e:
                    logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")

            results[timeframe] = timeframe_results

            # 保存到緩存
            self._save_to_cache(cache_key, timeframe_results)

        return results

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """重採樣數據到指定時間框架"""
        if timeframe == "1m":
            # 如果數據已經是分鐘級別，直接返回
            return data

        # 定義重採樣規則
        freq_map = {
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W",
            "1M": "1MS",
        }

        if timeframe not in freq_map:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return None

        try:
            # 確保索引是 datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data.index = pd.to_datetime(data.index)

            # OHLCV 重採樣規則
            agg_rules = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

            # 只保留存在的列
            available_rules = {col: rule for col, rule in agg_rules.items() if col in data.columns}

            resampled = data.resample(freq_map[timeframe]).agg(available_rules)

            # 移除空數據
            resampled = resampled.dropna()

            return resampled

        except Exception as e:
            logger.error(f"Error resampling data to {timeframe}: {e}")
            return None

    def _get_cache_key(self, symbol: str, timeframe: str, data: pd.DataFrame) -> str:
        """生成緩存鍵"""
        if not self.config.cache_results:
            return None

        # 創建數據指紋
        data_hash = hashlib.md5(
            f"{data.index[0]}_{data.index[-1]}_{len(data)}_{data['close'].iloc[-1]}".encode()
        ).hexdigest()[:16]

        return f"{symbol}_{timeframe}_{data_hash}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """從緩存載入結果"""
        if not cache_key or not self.config.cache_results:
            return None

        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"

        try:
            if cache_file.exists():
                # 檢查文件年齡（1小時過期）
                if time.time() - cache_file.stat().st_mtime < 3600:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            logger.debug(f"Cache load error: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """保存結果到緩存"""
        if not cache_key or not self.config.cache_results:
            return

        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.debug(f"Cache save error: {e}")

    def calculate_signals(
        self,
        indicators_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        signal_config: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        基於指標數據生成交易信號

        Args:
            indicators_data: 指標數據 {symbol: {timeframe: {indicator: data}}}
            signal_config: 信號配置

        Returns:
            {symbol: {timeframe: signals_df}} 信號字典
        """
        signal_config = signal_config or {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "cci_oversold": -100,
            "cci_overbought": 100,
            "williams_oversold": -80,
            "williams_overbought": -20,
        }

        signals = {}

        for symbol, timeframes_data in indicators_data.items():
            signals[symbol] = {}

            for timeframe, indicators in timeframes_data.items():
                try:
                    timeframe_signals = self._calculate_timeframe_signals(indicators, signal_config)
                    signals[symbol][timeframe] = timeframe_signals
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")

        return signals

    def _calculate_timeframe_signals(
        self, indicators: Dict[str, pd.DataFrame], config: Dict
    ) -> pd.DataFrame:
        """計算特定時間框架的信號"""
        if not indicators:
            return pd.DataFrame()

        # 獲取基準索引
        base_index = None
        for indicator_data in indicators.values():
            if isinstance(indicator_data, pd.DataFrame) and not indicator_data.empty:
                base_index = indicator_data.index
                break

        if base_index is None:
            return pd.DataFrame()

        signals = pd.DataFrame(index=base_index)

        # RSI 信號
        if "RSI_14" in indicators:
            rsi_data = indicators["RSI_14"]
            if isinstance(rsi_data, pd.DataFrame) and "RSI_14" in rsi_data.columns:
                rsi = rsi_data["RSI_14"]
                signals["rsi_buy"] = rsi < config["rsi_oversold"]
                signals["rsi_sell"] = rsi > config["rsi_overbought"]

        # MACD 信號
        if "MACD" in indicators:
            macd_data = indicators["MACD"]
            if isinstance(macd_data, pd.DataFrame):
                if "macd" in macd_data.columns and "signal" in macd_data.columns:
                    macd_line = macd_data["macd"]
                    signal_line = macd_data["signal"]
                    signals["macd_buy"] = (macd_line > signal_line) & (
                        macd_line.shift(1) <= signal_line.shift(1)
                    )
                    signals["macd_sell"] = (macd_line < signal_line) & (
                        macd_line.shift(1) >= signal_line.shift(1)
                    )

        # CCI 信號
        if "CCI_20" in indicators:
            cci_data = indicators["CCI_20"]
            if isinstance(cci_data, pd.DataFrame) and "CCI_20" in cci_data.columns:
                cci = cci_data["CCI_20"]
                signals["cci_buy"] = cci < config["cci_oversold"]
                signals["cci_sell"] = cci > config["cci_overbought"]

        # 布林帶信號
        if "BollingerBands" in indicators:
            bb_data = indicators["BollingerBands"]
            if isinstance(bb_data, pd.DataFrame):
                if all(col in bb_data.columns for col in ["upper_band", "lower_band", "percent_b"]):
                    signals["bb_buy"] = bb_data["percent_b"] < 0  # 價格低於下軌
                    signals["bb_sell"] = bb_data["percent_b"] > 1  # 價格高於上軌

        # 綜合信號
        buy_columns = [col for col in signals.columns if col.endswith("_buy")]
        sell_columns = [col for col in signals.columns if col.endswith("_sell")]

        if buy_columns:
            signals["total_buy_signals"] = signals[buy_columns].sum(axis=1)
        if sell_columns:
            signals["total_sell_signals"] = signals[sell_columns].sum(axis=1)

        # 最終信號
        signals["signal_strength"] = signals.get("total_buy_signals", 0) - signals.get(
            "total_sell_signals", 0
        )

        signals["final_signal"] = np.where(
            signals["signal_strength"] > 1,
            "BUY",
            np.where(signals["signal_strength"] < -1, "SELL", "HOLD"),
        )

        return signals

    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        return {
            "total_calculations": self.stats["total_calculations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_calculations"]),
            "total_time": self.stats["total_time"],
            "stocks_processed": self.stats["stocks_processed"],
            "avg_time_per_stock": self.stats["total_time"] / max(1, self.stats["stocks_processed"]),
        }

    def clear_cache(self):
        """清空緩存"""
        if self.config.cache_results:
            cache_dir = Path(self.config.cache_dir)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("Cache cleared")

    def optimize_parameters(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        parameter_ranges: Dict[str, List],
        metric: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        """
        優化指標參數

        Args:
            data: 測試數據
            indicator_name: 指標名稱
            parameter_ranges: 參數範圍 {'period': [10, 15, 20, 25]}
            metric: 優化目標指標

        Returns:
            最佳參數和性能
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        best_params = None
        best_score = float("-inf")
        results = []

        # 生成參數組合
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        from itertools import product

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            try:
                # 創建指標實例
                indicator_class = type(self.indicators[indicator_name])
                indicator = indicator_class(**params)

                # 計算指標
                indicator_result = indicator.calculate(data)
                signals = indicator.get_signals(data)

                # 計算性能指標
                performance = self._calculate_performance(data, signals, metric)

                results.append({"params": params, "score": performance})

                if performance > best_score:
                    best_score = performance
                    best_params = params

            except Exception as e:
                logger.debug(f"Parameter optimization error for {params}: {e}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": sorted(results, key=lambda x: x["score"], reverse=True),
        }

    def _calculate_performance(
        self, data: pd.DataFrame, signals: pd.DataFrame, metric: str
    ) -> float:
        """計算策略性能"""
        if "buy" not in signals.columns or "sell" not in signals.columns:
            return 0

        # 簡化的性能計算
        position = 0
        returns = []

        for i in range(1, len(signals)):
            if signals["buy"].iloc[i] and position == 0:
                position = 1
                entry_price = data["close"].iloc[i]
            elif signals["sell"].iloc[i] and position == 1:
                position = 0
                exit_price = data["close"].iloc[i]
                ret = (exit_price - entry_price) / entry_price
                returns.append(ret)

        if not returns:
            return 0

        returns = np.array(returns)

        if metric == "sharpe_ratio":
            if returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)
        elif metric == "total_return":
            return np.sum(returns)
        elif metric == "win_rate":
            return np.mean(returns > 0)
        else:
            return np.mean(returns)


def create_test_data(n_stocks: int = 100, n_periods: int = 1000) -> Dict[str, pd.DataFrame]:
    """創建測試數據"""
    stocks_data = {}

    for i in range(n_stocks):
        symbol = f"TEST{i:03d}"

        # 生成隨機價格數據
        dates = pd.date_range(start="2023-01-01", periods=n_periods, freq="1min")

        price = 100 + np.cumsum(np.random.randn(n_periods) * 0.01)
        high = price + np.random.exponential(0.5, n_periods)
        low = price - np.random.exponential(0.5, n_periods)
        volume = np.random.randint(1000, 10000, n_periods)

        stocks_data[symbol] = pd.DataFrame(
            {
                "open": price + np.random.randn(n_periods) * 0.1,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            },
            index=dates,
        )

    return stocks_data


if __name__ == "__main__":
    print("Technical Indicators Calculator - Stage 3 Implementation")
    print("=" * 60)

    # 創建測試數據
    print("Generating test data...")
    test_data = create_test_data(n_stocks=10, n_periods=500)

    # 初始化計算器
    config = CalculationConfig(
        timeframes=["5m", "15m", "1h"], batch_size=5, use_multiprocessing=False  # 測試時關閉多進程
    )

    calculator = IndicatorCalculator(config)

    # 計算指標
    print(f"Calculating indicators for {len(test_data)} stocks...")
    start_time = time.time()

    results = calculator.calculate_all_indicators(test_data)

    calculation_time = time.time() - start_time

    # 顯示結果
    print(f"\nCalculation completed in {calculation_time:.2f} seconds")
    print(f"Performance stats: {calculator.get_performance_stats()}")

    # 生成信號
    print("\nGenerating trading signals...")
    signals = calculator.calculate_signals(results)

    # 顯示示例結果
    if results:
        sample_symbol = list(results.keys())[0]
        sample_timeframe = list(results[sample_symbol].keys())[0]
        sample_indicators = list(results[sample_symbol][sample_timeframe].keys())

        print(f"\nSample results for {sample_symbol} ({sample_timeframe}):")
        print(f"Available indicators: {sample_indicators}")

        if signals and sample_symbol in signals:
            sample_signals = signals[sample_symbol].get(sample_timeframe)
            if sample_signals is not None and not sample_signals.empty:
                signal_counts = sample_signals["final_signal"].value_counts()
                print(f"Signal distribution: {signal_counts.to_dict()}")

    print("\n✓ Indicator Calculator ready for production use!")
