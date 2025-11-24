from __future__ import annotations

from typing import Iterable, Optional, Dict, List

import os
import time
from pathlib import Path

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
        local_root: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = float(request_timeout)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        env_local = os.environ.get("RL3_BINANCE_ROOT") or os.environ.get("RL3_DATA_ROOT")
        self.local_root = Path(local_root).expanduser().resolve() if local_root else (Path(env_local).expanduser().resolve() if env_local else None)

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

    @staticmethod
    def _parquet_tag(timeframe: str) -> str:
        token = timeframe.lower()
        if token in {"1min", "1m"}:
            return "1m"
        if token in {"5min", "5m"}:
            return "5m"
        if token in {"60m", "1h"}:
            return "1h"
        if token in {"1d", "day", "daily"}:
            return "1d"
        return token

    def _local_symbol_candidates(self, symbol: str) -> List[str]:
        tokens = {
            symbol,
            symbol.replace("-", ""),
            symbol.replace("-", "").replace("/", ""),
            symbol.replace("-", "").upper(),
            symbol.replace("-", "").upper().replace("/", ""),
        }
        mapped = self._map_symbol(symbol)
        tokens.add(mapped)
        tokens.add(mapped.replace("USDT", "USD"))
        tokens.add(mapped.replace("USDT", ""))
        return [t for t in {tok.upper() for tok in tokens if tok}]

    def _load_local(self, symbol: str, timeframe: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if self.local_root is None:
            return None

        tag = self._parquet_tag(timeframe)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        margin = pd.Timedelta(hours=6)
        t0 = start_ts - margin
        t1 = end_ts + margin
        ts_candidates = [
            "ts_utc",
            "timestamp",
            "time",
            "ts",
            "datetime",
            "open_time",
            "opentime",
            "kline_open_time",
            "openTime",
        ]

        for candidate in self._local_symbol_candidates(symbol):
            path = self.local_root / candidate / f"{tag}.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            cols_lower = {c.lower(): c for c in df.columns}
            ts_col = None
            for name in ts_candidates:
                if name in df.columns:
                    ts_col = name
                    break
                lower = name.lower()
                if lower in cols_lower:
                    ts_col = cols_lower[lower]
                    break
            if ts_col is None:
                if df.index.name:
                    ts_col = df.index.name
                    df = df.reset_index()
                else:
                    continue
            ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.assign(__ts=ts).dropna(subset=["__ts"])
            df = df.sort_values("__ts")
            df = df.loc[(df["__ts"] >= t0) & (df["__ts"] <= t1)]
            if df.empty:
                continue

            rename_map = {}
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    rename_map[col] = col
                    continue
                lower = col.lower()
                if lower in cols_lower:
                    rename_map[cols_lower[lower]] = col
                else:
                    alt = [c for c in df.columns if c.lower() == lower]
                    if alt:
                        rename_map[alt[0]] = col
            frame = df.rename(columns=rename_map)
            missing = [col for col in _OHLCV if col not in frame.columns]
            if missing:
                continue
            frame = frame.set_index("__ts")
            frame = frame[_OHLCV].astype(float)
            return frame
        return None

    def get_bars(
        self, symbol: str, start: str, end: str, timeframe: str = "5min"
    ) -> pd.DataFrame:
        local = self._load_local(symbol, timeframe, start, end)
        if local is not None and not local.empty:
            return local

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

        parquet_tag = {"1m": "1min", "5m": "5min", "1h": "60m", "1d": "1d"}.get(
            interval, interval
        )
        df.to_parquet(self._cache_path(symbol, parquet_tag))
        return df

    def fetch_ohlcv(
        self, symbols: Iterable[str], start: str, end: str, timeframe: str = "5min"
    ) -> Dict[str, pd.DataFrame]:
        return {
            symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols
        }


__all__ = ["BinanceBackend"]
