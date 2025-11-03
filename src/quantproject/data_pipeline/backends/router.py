from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from .alpha_vantage import AlphaVantageBackend
from .binance import BinanceBackend
from .yfinance import YFinanceBackend


_COLS = ["open", "high", "low", "close", "volume"]
_CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "BTCUSD", "ETHUSD"}


def _minutes_from_timeframe(timeframe: str) -> int | None:
    if not timeframe:
        return None
    tf = timeframe.strip().lower()
    if tf in {"1d", "daily", "day"}:
        return 1440
    if tf.endswith("min"):
        token = tf[:-3]
    elif tf.endswith("m"):
        token = tf[:-1]
    elif tf.endswith("h"):
        token = tf[:-1]
        try:
            return int(float(token) * 60)
        except ValueError:
            return None
    else:
        token = tf
    try:
        return int(float(token))
    except ValueError:
        return None


class DataRouter:
    """Route OHLCV requests to the best available backend per asset."""

    def __init__(
        self,
        *,
        yahoo: YFinanceBackend | None = None,
        alphavantage: AlphaVantageBackend | None = None,
        binance: BinanceBackend | None = None,
    ) -> None:
        self.yahoo = yahoo or YFinanceBackend()
        self.alpha = alphavantage or AlphaVantageBackend()
        self.binance = binance or BinanceBackend()

    def _providers_for(self, symbol: str, timeframe: str) -> list:
        symbol_up = symbol.upper()
        if symbol_up in _CRYPTO_SYMBOLS:
            return [self.binance, self.yahoo]

        has_alpha = bool(getattr(self.alpha, "api_key", None))
        minutes = _minutes_from_timeframe(timeframe)
        if has_alpha and (minutes is None or minutes >= 60):
            return [self.alpha, self.yahoo]

        providers = [self.yahoo]
        if self.alpha not in providers:
            providers.append(self.alpha)
        return providers

    def get_bars(
        self, symbol: str, start: str, end: str, timeframe: str = "5min"
    ) -> pd.DataFrame:
        for backend in self._providers_for(symbol, timeframe):
            df = backend.get_bars(symbol, start, end, timeframe)
            if df is not None and not df.empty:
                ordered = df.sort_index()
                ordered = ordered[~ordered.index.duplicated(keep="last")]
                if isinstance(ordered.columns, pd.MultiIndex):
                    ordered.columns = [
                        str(level).lower()
                        for level in ordered.columns.get_level_values(0)
                    ]
                missing = [col for col in _COLS if col not in ordered.columns]
                if missing:
                    continue
                return ordered[_COLS].copy()
        return pd.DataFrame(columns=_COLS)

    def fetch_ohlcv(
        self, symbols: Iterable[str], start: str, end: str, timeframe: str = "5min"
    ) -> Dict[str, pd.DataFrame]:
        return {
            symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols
        }


__all__ = ["DataRouter"]
