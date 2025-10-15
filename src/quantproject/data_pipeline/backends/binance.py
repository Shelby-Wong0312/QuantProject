from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

from .base import IDataBackend


_OHLCV = ["open", "high", "low", "close", "volume"]
_INTERVAL_MAP: Dict[str, tuple[str, int]] = {
    "1min": ("1m", 60_000),
    "1m": ("1m", 60_000),
    "5min": ("5m", 5 * 60_000),
    "5m": ("5m", 5 * 60_000),
    "60m": ("1h", 60 * 60_000),
    "1h": ("1h", 60 * 60_000),
    "1d": ("1d", 24 * 60 * 60_000),
}
_SYMBOL_FALLBACK = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}


class BinanceBackend(IDataBackend):
    """Binance OHLCV loader (no API key required)."""

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        cache_dir: str | Path = "./data/cache/parquet",
        *,
        request_timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = float(request_timeout)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        mapped = _SYMBOL_FALLBACK.get(symbol.upper())
        if mapped:
            return mapped
        s = symbol.upper().replace("-", "").replace("/", "")
        if s.endswith("USD") and not s.endswith("USDT"):
            s = s[:-3] + "USDT"
        return s

    @staticmethod
    def _interval(timeframe: str) -> tuple[str, int]:
        return _INTERVAL_MAP.get(timeframe.lower(), ("5m", 5 * 60_000))

    def _cache_path(self, symbol: str, interval_tag: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"BINANCE_{safe_symbol}_{interval_tag}.parquet"

    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = "5min") -> pd.DataFrame:
        mapped_symbol = self._map_symbol(symbol)
        interval, step_ms = self._interval(timeframe)

        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        if end_ts <= start_ts:
            return pd.DataFrame(columns=_OHLCV)

        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        frames: list[pd.DataFrame] = []
        current = start_ms
        limit = 1000

        for _ in range(200):
            params = {
                "symbol": mapped_symbol,
                "interval": interval,
                "limit": limit,
                "startTime": current,
                "endTime": end_ms,
            }
            try:
                resp = self.session.get(
                    f"{self.base_url}/api/v3/klines",
                    params=params,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except Exception:
                break

            payload = resp.json()
            if not payload:
                break

            batch_rows = []
            for entry in payload:
                open_time = int(entry[0])
                if open_time > end_ms:
                    continue
                batch_rows.append(
                    {
                        "open_time": pd.Timestamp(open_time, unit="ms", tz="UTC"),
                        "open": float(entry[1]),
                        "high": float(entry[2]),
                        "low": float(entry[3]),
                        "close": float(entry[4]),
                        "volume": float(entry[5]),
                    }
                )

            if batch_rows:
                frames.append(pd.DataFrame(batch_rows))

            last_open = int(payload[-1][0])
            next_start = last_open + step_ms
            if next_start <= current:
                next_start = current + step_ms
            current = next_start

            if current >= end_ms:
                break
            if len(payload) < limit:
                break
            time.sleep(0.2)

        if not frames:
            return pd.DataFrame(columns=_OHLCV)

        df = pd.concat(frames, axis=0, ignore_index=True)
        if df.empty:
            return pd.DataFrame(columns=_OHLCV)

        df = df.drop_duplicates(subset=["open_time"], keep="last")
        df = df.sort_values("open_time").set_index("open_time")
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df[_OHLCV].copy().dropna()
        if df.empty:
            return pd.DataFrame(columns=_OHLCV)

        parquet_tag = {"1m": "1min", "5m": "5min", "1h": "60m", "1d": "1d"}.get(interval, interval)
        df.to_parquet(self._cache_path(symbol, parquet_tag))
        return df

    def fetch_ohlcv(
        self, symbols: Iterable[str], start: str, end: str, timeframe: str = "5min"
    ) -> Dict[str, pd.DataFrame]:
        return {symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols}


__all__ = ["BinanceBackend"]
