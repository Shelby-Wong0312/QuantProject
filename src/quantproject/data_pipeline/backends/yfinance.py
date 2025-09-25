from __future__ import annotations

from typing import Dict, Iterable, Optional

from pathlib import Path

import pandas as pd
import yfinance as yf


_COLS = ["open", "high", "low", "close", "volume"]
_TF_MAP = {
    "1min": "1m",
    "5min": "5m",
    "1h": "60m",
    "1d": "1d",
}
_MAX_BACK = {
    "1m": pd.Timedelta(days=7),
    "5m": pd.Timedelta(days=60),
    "60m": pd.Timedelta(days=730),
    "1d": pd.Timedelta(days=7300),
}
_DEFAULT_MAX = pd.Timedelta(days=60)
_SHRUNK_5M = pd.Timedelta(days=55)
_MIN_DELTA = pd.Timedelta(minutes=5)


def _normalize_timeframe(timeframe: str) -> str:
    return _TF_MAP.get((timeframe or "").lower(), "5m")


def _parse_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _ensure_end_after_start(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Timestamp:
    if end_ts <= start_ts:
        return start_ts + _MIN_DELTA
    return end_ts


class YFinanceBackend:
    """Thin wrapper around yfinance to fetch OHLCV data."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("data_cache") / "yfinance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = symbol.replace('/', '_').replace(' ', '_')
        return self.cache_dir / f"{safe_symbol}_{interval}.parquet"


    def get_bars(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        yf_interval = _normalize_timeframe(interval)

        start_ts = _parse_ts(start)
        end_ts = _parse_ts(end)

        if start_ts is None and end_ts is not None:
            start_ts = end_ts - _DEFAULT_MAX
        if end_ts is None and start_ts is not None:
            end_ts = start_ts + _DEFAULT_MAX
        if start_ts is None and end_ts is None:
            end_ts = pd.Timestamp.utcnow().tz_localize("UTC")
            start_ts = end_ts - _DEFAULT_MAX

        end_ts = _ensure_end_after_start(start_ts, end_ts)

        max_back = _MAX_BACK.get(yf_interval, _DEFAULT_MAX)
        span = end_ts - start_ts
        if span > max_back:
            yf_interval = "5m"
            span = _SHRUNK_5M
            start_ts = end_ts - span

        start_arg = start_ts.tz_convert(None)
        end_arg = end_ts.tz_convert(None)

        try:
            data = yf.download(
                symbol,
                start=start_arg,
                end=end_arg,
                interval=yf_interval,
                progress=False,
                threads=False,
                auto_adjust=False,
            )
        except Exception:
            data = None

        if data is None or data.empty:
            return pd.DataFrame(columns=_COLS)

        df = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        if not all(col in df.columns for col in _COLS):
            return pd.DataFrame(columns=_COLS)

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[_COLS].copy().dropna()
        if df.empty:
            return pd.DataFrame(columns=_COLS)

        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float)

        cache_interval = "5min" if yf_interval == "5m" else yf_interval
        df.to_parquet(self._cache_path(symbol, cache_interval))

        return df

    def fetch_ohlcv(self, symbols: Iterable[str], start: str, end: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frames[symbol] = self.get_bars(symbol, start, end, interval)
        return frames


__all__ = ["YFinanceBackend"]
