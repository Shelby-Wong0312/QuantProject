from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable

import pandas as pd


class IDataBackend(ABC):
    """Common interface for OHLCV data backends."""

    @abstractmethod
    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = "5min") -> pd.DataFrame:
        """Return OHLCV bars for the window; empty DataFrame if unavailable."""

    def fetch_ohlcv(
        self, symbols: Iterable[str], start: str, end: str, timeframe: str = "5min"
    ) -> Dict[str, pd.DataFrame]:
        return {symbol: self.get_bars(symbol, start, end, timeframe) for symbol in symbols}


__all__ = ["IDataBackend"]
