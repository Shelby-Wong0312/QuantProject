from __future__ import annotations

import os
from typing import Dict, Iterable

from pathlib import Path

import pandas as pd
import requests

from .base import IDataBackend


_COLS = ["open", "high", "low", "close", "volume"]
_INTRADAY_INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "60m": "60min",
}


class AlphaVantageBackend(IDataBackend):
    """Lightweight AlphaVantage OHLCV loader (skips if no API key)."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        api_key: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path("data_cache") / "alphavantage"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.timeout = float(request_timeout)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_symbol}_{interval}.parquet"

    def _intraday(
        self, symbol: str, interval: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> pd.DataFrame:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            resp.json()
        except Exception:
            return pd.DataFrame(columns=_COLS)

        key = next((k for k in data.keys() if "Time Series" in k), None)
        if not key:
            return pd.DataFrame(columns=_COLS)

        rows = []
        for timestamp, values in data[key].items():
            ts = pd.Timestamp(timestamp, tz="UTC")
            if ts < start_ts or ts > end_ts:
                continue
            try:
                rows.append(
                    {
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": float(values.get("5. volume", 0.0)),
                        "open_time": ts,
                    }
                )
            except KeyError:
                continue
        if not rows:
            return pd.DataFrame(columns=_COLS)
        df = pd.DataFrame(rows).set_index("open_time").sort_index()
        return df

    def _daily(
        self, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> pd.DataFrame:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            resp.json()
        except Exception:
            return pd.DataFrame(columns=_COLS)

        key = next((k for k in data.keys() if "Time Series" in k), None)
        if not key:
            return pd.DataFrame(columns=_COLS)

        rows = []
        for date, values in data[key].items():
            ts = pd.Timestamp(date, tz="UTC")
            if ts < start_ts or ts > end_ts:
                continue
            try:
                rows.append(
                    {
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": float(values.get("6. volume", 0.0)),
                        "open_time": ts,
                    }
                )
            except KeyError:
                continue
        if not rows:
            return pd.DataFrame(columns=_COLS)
        df = pd.DataFrame(rows).set_index("open_time").sort_index()
        return df

    def get_bars(
        self, symbol: str, start: str, end: str, timeframe: str = "5min"
    ) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame(columns=_COLS)

        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        if end_ts <= start_ts:
            return pd.DataFrame(columns=_COLS)

        if timeframe.lower() in _INTRADAY_INTERVALS:
            interval = _INTRADAY_INTERVALS[timeframe.lower()]
            df = self._intraday(symbol, interval, start_ts, end_ts)
        else:
            df = self._daily(symbol, start_ts, end_ts)

        if df.empty:
            return df.reindex(columns=_COLS)

        df = df[_COLS].copy().dropna()
        if df.empty:
            return pd.DataFrame(columns=_COLS)

        df.to_parquet(self._cache_path(symbol, timeframe.lower()))
        return df

    def fetch_ohlcv(
        self, symbols: Iterable[str], start: str, end: str, timeframe: str = "5min"
    ) -> Dict[str, pd.DataFrame]:
        return {
            symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols
        }


__all__ = ["AlphaVantageBackend"]
