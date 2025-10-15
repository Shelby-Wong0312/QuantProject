from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests
import yfinance as yf

from .base import IDataBackend

_COLS = ["open", "high", "low", "close", "volume"]
_PERIOD_FALLBACK = {"1m": "7d", "5m": "60d", "60m": "730d", "1d": "10y"}
_MAX_BACK = {
    "1m": pd.Timedelta(days=7),
    "5m": pd.Timedelta(days=60),
    "60m": pd.Timedelta(days=730),
    "1d": pd.Timedelta(days=3650),
}
_SHRUNK_5M = pd.Timedelta(days=55)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.copy()
    if isinstance(base.columns, pd.MultiIndex):
        raw = base.columns.get_level_values(0)
    else:
        raw = base.columns
    normalized = []
    for col in raw:
        name = str(col).strip().lower().replace(" ", "_")
        normalized.append(name)
    base.columns = normalized
    base = base.loc[:, ~pd.Index(base.columns).duplicated()]
    allowed = set(_COLS) | {"adj_close"}
    keep = [col for col in base.columns if col in allowed]
    base = base.loc[:, keep]
    if "adj_close" in base.columns:
        if "close" in base.columns:
            base = base.drop(columns=["adj_close"])
        else:
            base = base.rename(columns={"adj_close": "close"})
    return base


def _prepare_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class YFinanceBackend(IDataBackend):
    """YFinance OHLCV loader with start/end retries and period fallback."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else Path("data_cache") / "yfinance"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_symbol}_{interval}.parquet"

    def _download_via_api(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
        headers = {"User-Agent": "Mozilla/5.0"}
        params_template = {"interval": interval, "includePrePost": "false"}

        for _ in range(3):
            params = dict(params_template)
            params["period1"] = int(start_ts.timestamp())
            params["period2"] = int(end_ts.timestamp())
            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=15)
            except Exception:
                continue
            if resp.status_code == 429:
                continue
            try:
                payload = resp.json()
            except Exception:
                continue
            result = payload.get("chart", {}).get("result")
            if not result:
                continue
            result[0]
            timestamps = data.get("timestamp")
            quotes = data.get("indicators", {}).get("quote") or []
            if not timestamps or not quotes:
                continue
            quote = quotes[0]
            frame = pd.DataFrame({col: quote.get(col) for col in _COLS})
            frame.index = pd.to_datetime(timestamps, unit="s", utc=True)
            frame = frame[_COLS].copy().dropna()
            if frame.empty:
                continue
            for col in ("open", "high", "low", "close"):
                frame[col] = frame[col].astype(float)
            frame["volume"] = frame["volume"].fillna(0.0).astype(float)
            frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
            if frame.empty:
                continue
            return frame
        return None

    def _load_cache(
        self, cache_path: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        if not cache_path.exists():
            return None
        try:
            cached = pd.read_parquet(cache_path)
        except Exception:
            return None
        cached = _normalize_columns(cached)
        cached.index = pd.to_datetime(cached.index, utc=True)
        cached = cached.sort_index()
        cached = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
        missing = [col for col in _COLS if col not in cached.columns]
        if missing:
            return None
        cached = cached[_COLS].copy().dropna()
        if cached.empty:
            return None
        for col in ("open", "high", "low", "close"):
            cached[col] = cached[col].astype(float)
        cached["volume"] = cached["volume"].astype(float)
        return cached

    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = "5min") -> pd.DataFrame:
        tf_map = {
            "1min": "1m",
            "1m": "1m",
            "5min": "5m",
            "5m": "5m",
            "60m": "60m",
            "1h": "60m",
            "1d": "1d",
        }
        interval = tf_map.get(timeframe, "5m")
        parquet_tag = {"1m": "1min", "5m": "5min", "60m": "60m", "1d": "1d"}.get(interval, interval)
        cache_path = self._cache_path(symbol, parquet_tag)

        end_ts = _prepare_timestamp(pd.Timestamp(end, tz="UTC"))
        start_ts = _prepare_timestamp(pd.Timestamp(start, tz="UTC"))
        if end_ts <= start_ts:
            return pd.DataFrame(columns=_COLS)

        span = end_ts - start_ts
        limit = _MAX_BACK.get(interval, pd.Timedelta(days=60))
        if span > limit:
            shrink = _SHRUNK_5M if interval == "5m" else limit
            start_ts = end_ts - shrink

        df: Optional[pd.DataFrame] = None

        for attempt, backoff in enumerate((0.0, 2.0, 5.0), 1):
            jitter = random.random() * 0.5
            delay = backoff + jitter
            if delay > 0.0:
                time.sleep(delay)
            delay = backoff + (random.random() if backoff else 0.0)
            if delay > 0.0:
                time.sleep(delay)
            try:
                candidate = yf.download(
                    tickers=symbol,
                    start=start_ts.tz_convert(None),
                    end=end_ts.tz_convert(None),
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception:
                continue
            if candidate is None or candidate.empty:
                continue
            df = candidate
            break

        if df is None:
            df = self._download_via_api(symbol, start_ts, end_ts, interval)

        if df is None:
            cached = self._load_cache(cache_path, start_ts, end_ts)
            if cached is not None:
                return cached
            period = _PERIOD_FALLBACK.get(interval, "60d")
            for attempt, backoff in enumerate((0.0, 2.0, 5.0), 1):
                jitter = random.random() * 0.5
                delay = backoff + jitter
                if delay > 0.0:
                    time.sleep(delay)
                delay = backoff + (random.random() if backoff else 0.0)
                if delay > 0.0:
                    time.sleep(delay)
                try:
                    candidate = yf.download(
                        tickers=symbol,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
                except Exception:
                    continue
                if candidate is None or candidate.empty:
                    continue
                df = candidate
                break
            if df is None:
                return pd.DataFrame(columns=_COLS)

        df = _normalize_columns(df)
        if not all(col in df.columns for col in _COLS):
            return pd.DataFrame(columns=_COLS)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df[_COLS].copy().dropna()
        if df.empty:
            cached = self._load_cache(cache_path, start_ts, end_ts)
            return cached if cached is not None else pd.DataFrame(columns=_COLS)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float)
        df.to_parquet(cache_path)
        return df

    def fetch_ohlcv(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        timeframe: str = "5min",
    ) -> Dict[str, pd.DataFrame]:
        return {symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols}


__all__ = ["YFinanceBackend"]
