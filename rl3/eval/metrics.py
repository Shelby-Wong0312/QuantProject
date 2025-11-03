from __future__ import annotations

import math
import statistics as stats
from typing import Any, Dict, Iterable, List

_CRYPTO_KEYWORDS = {"BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "USDT", "USDC"}


def _is_crypto_symbol(symbol: str) -> bool:
    token = symbol.upper().strip()
    if not token or token.endswith("=X"):
        return False
    return any(keyword in token for keyword in _CRYPTO_KEYWORDS)


def _resolve_calendar(symbols: Iterable[str], calendar: str | None) -> str:
    if calendar is None:
        symbols_list = list(symbols or [])
        if symbols_list and all(_is_crypto_symbol(sym) for sym in symbols_list):
            return "crypto"
        return "equity"
    calendar_key = calendar.lower()
    if calendar_key not in {"equity", "crypto"}:
        raise ValueError(f"Unsupported calendar '{calendar}'. Use 'equity' or 'crypto'.")
    return calendar_key


def _timeframe_minutes(tf: str | None) -> float | None:
    if not tf:
        return None
    token = str(tf).strip().lower()
    if not token:
        return None
    shortcuts = {"1d": None, "d": None, "day": None, "daily": None}
    if token in shortcuts:
        return None
    if token.endswith("min"):
        token = token[:-3]
        try:
            return float(token)
        except ValueError:
            return None
    if token.endswith("m"):
        base = token[:-1]
        try:
            return float(base)
        except ValueError:
            return None
    if token.endswith("h"):
        base = token[:-1]
        try:
            return float(base) * 60.0
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def _ann_factor(timeframe: str, symbols: Iterable[str], calendar: str | None) -> float:
    tf = (timeframe or "5min").lower()
    cal = _resolve_calendar(symbols, calendar)
    is_crypto = cal == "crypto"

    if tf in ("1d", "d", "day", "1day", "daily"):
        return 365.0 if is_crypto else 252.0

    minutes = _timeframe_minutes(tf)
    if minutes is None or minutes <= 0:
        return 365.0 if is_crypto else 252.0

    if is_crypto:
        return (365.0 * 24.0 * 60.0) / minutes
    return (252.0 * 6.5 * 60.0) / minutes


def compute(
    returns: List[float],
    equity: List[float],
    timeframe: str = "5min",
    symbols: Iterable[str] | None = None,
    calendar: str | None = None,
) -> Dict[str, Any]:
    symbols_list = list(symbols or [])
    returns_list = list(returns) if returns else [0.0]

    mu = sum(returns_list) / len(returns_list)
    sigma = stats.pstdev(returns_list) if len(returns_list) > 1 else 0.0

    cal_key = _resolve_calendar(symbols_list, calendar)
    ann_factor = float(_ann_factor(timeframe, symbols_list, cal_key))

    ann_ret = mu * ann_factor
    ann_vol = sigma * math.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    max_dd, peak = 0.0, -1e18
    equity_series = list(equity) if equity else [1.0]
    for value in equity_series:
        v_eff = max(float(value), 1e-6)
        peak = max(peak, v_eff)
        drawdown = (v_eff - peak) / peak
        max_dd = min(max_dd, drawdown)
    max_dd = max(max_dd, -1.0)

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "returns_mean": float(mu),
        "returns_std": float(sigma),
        "calendar": cal_key,
    }


__all__ = ["compute"]
