from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset

from ..backends.router import DataRouter


def _load_with_router(
    symbols: Iterable[str], start: str, end: str, timeframe: str
) -> Dict[str, pd.DataFrame]:
    router = DataRouter()
    raw: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = router.get_bars(symbol, start, end, timeframe)
        if df is not None and not df.empty:
            raw[symbol] = df
    return raw


def _timeframe_minutes(timeframe: Optional[str]) -> Optional[int]:
    if not timeframe:
        return None
    tf = timeframe.strip().lower()
    if not tf:
        return None
    if tf in {"1d", "day", "daily"}:
        return 1440
    for suffix, multiplier in (("min", 1), ("m", 1), ("h", 60)):
        if tf.endswith(suffix):
            token = tf[: -len(suffix)] or "1"
            try:
                return int(float(token) * multiplier)
            except ValueError:
                return None
    try:
        return int(float(tf))
    except ValueError:
        return None


def _normalize_index(df: pd.DataFrame, minutes: Optional[int]) -> pd.DataFrame:
    if minutes is None:
        return df[~df.index.duplicated(keep="last")]
    freq = to_offset(f"{minutes}min")
    floored = pd.to_datetime(df.index, utc=True).floor(freq)
    if not floored.equals(df.index):
        df = df.copy()
        df.index = floored
    return df[~df.index.duplicated(keep="last")]


def _align_frames(
    raw: Dict[str, pd.DataFrame], timeframe: Optional[str]
) -> Dict[str, pd.DataFrame]:
    if not raw:
        return {}

    minutes = _timeframe_minutes(timeframe)
    normalized = {symbol: _normalize_index(df, minutes) for symbol, df in raw.items()}

    common_index = None
    for df in normalized.values():
        idx = pd.to_datetime(df.index, utc=True)
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) == 0:
        return {}
    return {
        symbol: normalized[symbol].reindex(common_index).dropna().copy()
        for symbol in normalized
    }


def load_and_align(
    symbols: List[str], start: str, end: str, timeframe: str = "5min"
) -> Dict[str, pd.DataFrame]:
    raw = _load_with_router(symbols, start, end, timeframe)
    if not raw:
        raise AssertionError(
            "No symbols returned data; check timeframe/start/end or connectivity."
        )
    aligned = _align_frames(raw, timeframe)
    if not aligned:
        raise AssertionError(
            "Unable to align data; check symbol coverage or timeframe overlap."
        )
    return aligned


def load_and_align_router(
    symbols: Iterable[str], start: str, end: str, timeframe: str = "5min"
) -> Dict[str, pd.DataFrame]:
    raw = _load_with_router(symbols, start, end, timeframe)
    return _align_frames(raw, timeframe)


__all__ = ["load_and_align", "load_and_align_router"]
