# quant_project/strategy/indicators.py
# FULLY EXPANDED VERSION

import pandas as pd
import pandas_ta as ta
import numpy as np

# --- 標準技術指標 ---


def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """一次性計算並添加所有需要的指標欄位到DataFrame中"""

    # MA (EMA for responsiveness)
    df["ema_short"] = ta.ema(df["Close"], length=params["ma_short_period"])
    df["ema_long"] = ta.ema(df["Close"], length=params["ma_long_period"])

    # BIAS
    bias_ma = ta.sma(df["Close"], length=params["bias_period"])
    df["bias"] = ((df["Close"] - bias_ma) / bias_ma) * 100

    # KD (Stochastic)
    stoch = ta.stoch(
        df["High"],
        df["Low"],
        df["Close"],
        k=params["kd_k"],
        d=params["kd_d"],
        smooth_k=params["kd_smooth"],
    )
    if stoch is not None and not stoch.empty:
        df = pd.concat([df, stoch], axis=1)

    # MACD
    macd = ta.macd(
        df["Close"],
        fast=params["macd_fast"],
        slow=params["macd_slow"],
        signal=params["macd_signal"],
    )
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)

    # RSI
    df["rsi"] = ta.rsi(df["Close"], length=params["rsi_period"])

    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=params["bb_period"], std=params["bb_std"])
    if bbands is not None and not bbands.empty:
        df = pd.concat([df, bbands], axis=1)

    # Ichimoku Cloud
    ichimoku, _ = ta.ichimoku(
        df["High"],
        df["Low"],
        df["Close"],
        tenkan=params["ichi_tenkan"],
        kijun=params["ichi_kijun"],
        senkou=params["ichi_senkou_b"],
    )
    if ichimoku is not None and not ichimoku.empty:
        df = pd.concat([df, ichimoku], axis=1)

    # Volume MA
    df["volume_ma"] = ta.sma(df["Volume"], length=params["vol_ma_period"])

    # ATR
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=params["atr_period"])

    return df


# --- 價格行為與理論分析 ---


def get_candlestick_patterns(df_slice: pd.DataFrame) -> dict:
    """檢測K線形態"""
    patterns = {}
    # 使用pandas_ta的內建功能來檢測所有形態
    # cdl_pattern會返回一個包含所有形態檢測結果的DataFrame
    pattern_df = ta.cdl_pattern(
        df_slice["Open"], df_slice["High"], df_slice["Low"], df_slice["Close"], name="all"
    )

    # 我們只取最新的（最後一根K線）的結果
    latest_patterns = pattern_df.iloc[-1]

    # 找出所有被觸發的形態 (值不為0)
    triggered = latest_patterns[latest_patterns != 0]

    # 看漲形態的值為正 (e.g., 100), 看跌為負 (e.g., -100)
    patterns["bullish"] = triggered[triggered > 0].index.to_list()
    patterns["bearish"] = triggered[triggered < 0].index.to_list()

    return patterns


def get_dow_theory_trend(df_slice: pd.DataFrame, lookback: int = 60) -> str:
    """簡化版道氏理論趨勢判斷"""
    if len(df_slice) < lookback:
        return "neutral"

    recent_data = df_slice.iloc[-lookback:]

    # 使用 find_peaks 找出顯著高低點
    from scipy.signal import find_peaks

    high_peaks_indices, _ = find_peaks(recent_data["High"], distance=5)
    low_peaks_indices, _ = find_peaks(-recent_data["Low"], distance=5)

    if len(high_peaks_indices) < 2 or len(low_peaks_indices) < 2:
        return "neutral"

    last_high = recent_data["High"].iloc[high_peaks_indices[-1]]
    prev_high = recent_data["High"].iloc[high_peaks_indices[-2]]

    last_low = recent_data["Low"].iloc[low_peaks_indices[-1]]
    prev_low = recent_data["Low"].iloc[low_peaks_indices[-2]]

    if last_high > prev_high and last_low > prev_low:
        return "uptrend"
    elif last_high < prev_high and last_low < prev_low:
        return "downtrend"

    return "neutral"
