"""Provider-facing helpers for retrieving OHLCV data via DataRouter."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..backends.router import DataRouter

_COLS = ["open", "high", "low", "close", "volume"]
_CRYPTO = {"BTC-USD", "ETH-USD", "BTCUSD", "ETHUSD"}

_PROVIDER_ATTR = {
    "yahoo": "yahoo",
    "alphavantage": "alpha",
    "binance": "binance",
}


def _is_crypto(symbol: str) -> bool:
    return symbol.upper() in _CRYPTO


def _is_fx(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith("=X") or "/" in s or (len(s) == 6 and s.isalpha())


def _provider_order(symbol: str, prefer: Optional[List[str]]) -> List[str]:
    if prefer:
        return prefer
    if _is_crypto(symbol):
        return ["binance", "yahoo", "alphavantage"]
    if _is_fx(symbol):
        return ["alphavantage", "yahoo"]
    return ["alphavantage", "yahoo"] if os.getenv("ALPHAVANTAGE_API_KEY") else ["yahoo"]


def fetch_bars(
    symbols: Iterable[str],
    start: str,
    end: str,
    timeframe: str = "5min",
    *,
    router: Optional[DataRouter] = None,
    prefer: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV frames for the requested symbols using provider order.

    Parameters
    ----------
    symbols: Iterable[str]
        Symbols to fetch.
    start, end: str
        Inclusive datetime boundaries understood by backends.
    timeframe: str
        Bar interval (e.g. "5min", "60m").
    router: Optional[DataRouter]
        Reuse an existing router instance (mainly for testing).
    prefer: Optional[List[str]]
        Override provider order (names: "binance", "alphavantage", "yahoo").
    """

    data_router = router or DataRouter()
    results: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        order = _provider_order(symbol, prefer)
        for provider_name in order:
            attr = _PROVIDER_ATTR.get(provider_name)
            if not attr:
                continue
            backend = getattr(data_router, attr, None)
            if backend is None:
                continue
            try:
                df = backend.get_bars(symbol, start, end, timeframe)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            ordered = df.sort_index()
            ordered = ordered[~ordered.index.duplicated(keep="last")]
            results[symbol] = (
                ordered[_COLS].copy()
                if all(col in ordered for col in _COLS)
                else ordered.copy()
            )
            break

    return results
