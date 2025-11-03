from __future__ import annotations

import time
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
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
            Path(cache_dir)
            if cache_dir is not None
            else Path("data_cache") / "yfinance"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.max_retries = getattr(self, 'max_retries', 3)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_symbol}_{interval}.parquet"

    def _download_via_api(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        import requests, time
        import pandas as pd
        import numpy as np
        from datetime import datetime, timezone

        if interval == "5min":
            interval = "5m"
        elif interval in ("60min", "1h"):
            interval = "60m"

        params = {
            "period1": int(start_ts),
            "period2": int(end_ts),
            "interval": interval,
            "events": "history",
            "includePrePost": "false",
        }
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Connection": "keep-alive",
        }

        last_err = None
        max_retry = getattr(self, "max_retries", 3)
        for attempt in range(max_retry):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=20)
                if resp.status_code >= 500 or resp.status_code == 429:
                    time.sleep(1.0 + attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                chart = data.get("chart", {})
                result = (chart.get("result") or [None])[0]
                if not result:
                    raise ValueError(f"Empty result for {symbol}")

                timestamps = result.get("timestamp")
                indicators = result.get("indicators", {})
                quote = (indicators.get("quote") or [None])[0]
                if not timestamps or not quote:
                    raise ValueError(f"Missing timestamp/quote for {symbol}")

                ts = pd.to_datetime(np.array(timestamps, dtype="int64"), unit="s", utc=True)
                df = pd.DataFrame(
                    {
                        "open": quote.get("open"),
                        "high": quote.get("high"),
                        "low": quote.get("low"),
                        "close": quote.get("close"),
                        "volume": quote.get("volume"),
                    },
                    index=ts,
                )

                df = df.dropna(how="all")
                if df.empty:
                    raise ValueError("All-NaN rows")

                df = df[~df.index.duplicated(keep="last")].sort_index()

                for col in ("open", "high", "low", "close"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
                df = df.dropna(subset=["open", "high", "low", "close"])
                if df.empty:
                    raise ValueError("No valid OHLC rows")

                start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                df = df[(df.index >= start_dt) & (df.index < end_dt)]
                if df.empty:
                    raise ValueError("No rows within requested window")

                return df[_COLS].copy()
            except Exception as err:
                last_err = err
                time.sleep(0.5 + attempt * 0.5)
                continue

        raise last_err if last_err else RuntimeError(f"Yahoo API failed for {symbol}")

    def _load_cache(
        self, cache_path: Path, start_ts: int, end_ts: int
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

    def get_bars(
        self, symbol: str, start: str, end: str, timeframe: str = "5min"
    ) -> pd.DataFrame:
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
        parquet_tag = {"1m": "1min", "5m": "5min", "60m": "60m", "1d": "1d"}.get(
            interval, interval
        )
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
            df = self._download_via_api(symbol, int(start_ts.timestamp()), int(end_ts.timestamp()), interval)

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
        return {
            symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols
        }


__all__ = ["YFinanceBackend"]
