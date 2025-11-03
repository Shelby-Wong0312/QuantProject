from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, List

FEATURES = globals().get("FEATURES", {})

FeatureFn = Callable[[pd.DataFrame], pd.Series]


class FeatureRegistry:
    def __init__(self) -> None:
        self._fns: Dict[str, FeatureFn] = {}

    def add(self, name: str, fn: FeatureFn) -> None:
        self._fns[name] = fn

    def build(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for name in features:
            if name == "close":
                if "close" not in df.columns:
                    raise KeyError("source dataframe missing 'close' column")
                frames.append(
                    pd.DataFrame(
                        {"close": pd.to_numeric(df["close"], errors="coerce")},
                        index=df.index,
                    )
                )
                continue
            fn = self._fns.get(name)
            if fn is None:
                raise KeyError(f"feature '{name}' not registered")
            result = fn(df)
            if isinstance(result, pd.Series):
                frames.append(result.to_frame(name=name))
            else:
                frames.append(result)
        if not frames:
            return pd.DataFrame(index=df.index)
        X = pd.concat(frames, axis=1)
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X = X.sort_index()
        X = X[~X.index.duplicated(keep="last")]
        return X


registry = FeatureRegistry()

def close_feature(df: pd.DataFrame) -> pd.Series:
    return _safe_close(df)


def register(name: str, fn: FeatureFn) -> None:
    FEATURES[name] = fn
    registry.add(name, fn)


def build(df: pd.DataFrame, feat_names: List[str]) -> pd.DataFrame:
    return registry.build(df, list(feat_names))

def _safe_close(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float)


def logret(df: pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    close = _safe_close(df)
    return np.log(np.clip(close, eps, None)).diff().fillna(0.0)


def rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    close = _safe_close(df)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / n
    avg_up = up.ewm(alpha=alpha, adjust=False).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-12)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(0.0)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    close = _safe_close(df)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    return line.fillna(0.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = _safe_close(df)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def vol_ewma(df: pd.DataFrame, span: int = 20) -> pd.Series:
    close = _safe_close(df)
    returns = close.pct_change().fillna(0.0)
    mean = returns.ewm(span=span, adjust=False).mean()
    variance = ((returns - mean) ** 2).ewm(span=span, adjust=False).mean()
    return np.sqrt(variance).fillna(0.0)


def vol_zscore(df: pd.DataFrame, n: int = 20) -> pd.Series:
    volume = df["volume"].astype(float)
    mean = volume.rolling(n, min_periods=1).mean()
    std = volume.rolling(n, min_periods=1).std().replace(0.0, np.nan)
    return ((volume - mean) / std).fillna(0.0)


def range_pct(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = _safe_close(df).replace(0.0, np.nan)
    return ((high - low) / close).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def trend_persist(df: pd.DataFrame, n: int = 10) -> pd.Series:
    close = _safe_close(df)
    returns = close.pct_change().fillna(0.0)
    signs = np.sign(returns)
    return (signs.rolling(n, min_periods=1).sum() / n).fillna(0.0)


def breakout_flags(df: pd.DataFrame, n: int = 20) -> pd.Series:
    close = _safe_close(df)
    highest = close.rolling(n, min_periods=1).max()
    lowest = close.rolling(n, min_periods=1).min()
    bullish = (close >= highest).astype(float)
    bearish = (close <= lowest).astype(float) * -1.0
    return (bullish + bearish).clip(-1.0, 1.0).fillna(0.0)


register("close", _safe_close)
register("logret", logret)
register("rsi", rsi)
register("macd", macd)
register("atr", atr)
register("vol_ewma", vol_ewma)
register("vol_zscore", vol_zscore)
register("range_pct", range_pct)
register("trend_persist", trend_persist)
register("breakout_flags", breakout_flags)
