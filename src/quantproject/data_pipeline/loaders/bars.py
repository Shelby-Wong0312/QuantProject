from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..backends.yfinance import YFinanceBackend

_REQ_COLS = ["open", "high", "low", "close", "volume"]


def _sanitize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    out = df.sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")

    if not set(_REQ_COLS).issubset(out.columns):
        return None

    out = out[_REQ_COLS].copy()
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    if out.empty:
        return None

    return out


def load_and_align(symbols: List[str], start: str, end: str, timeframe: str = "5min") -> Dict[str, pd.DataFrame]:
    backend = YFinanceBackend()
    raw: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        clean = _sanitize(backend.get_bars(symbol, start, end, timeframe))
        if clean is not None and not clean.empty:
            raw[symbol] = clean

    assert raw, "No symbols returned data; check timeframe/start/end or internet connectivity."

    common_index = None
    for frame in raw.values():
        idx = pd.to_datetime(frame.index, utc=True)
        common_index = idx if common_index is None else common_index.intersection(idx)

    data = {symbol: frame.reindex(common_index).dropna().copy() for symbol, frame in raw.items()}
    return data


__all__ = ["load_and_align"]
