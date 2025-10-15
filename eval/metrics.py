from __future__ import annotations

import math
import statistics as stats
from typing import Any, Dict, Iterable, List


_CRYPTO_KEYWORDS = {"BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "USDT", "USDC"}


def _is_crypto_symbol(symbol: str) -> bool:
    u = symbol.upper().strip()
    if not u or u.endswith("=X"):
        return False
    return any(keyword in u for keyword in _CRYPTO_KEYWORDS)


def _ann_factor(timeframe: str, symbols: Iterable[str]) -> float:
    tf = (timeframe or "5min").lower()
    symbols_list = list(symbols or [])
    is_crypto = bool(symbols_list) and all(
        _is_crypto_symbol(sym) for sym in symbols_list
    )

    if tf in ("5m", "5min"):
        return 365 * 24 * 60 / 5 if is_crypto else 252 * 6.5 * 60 / 5
    if tf in ("1m", "1min"):
        return 365 * 24 * 60 if is_crypto else 252 * 6.5 * 60
    if tf in ("60m", "1h", "1hour"):
        return 365 * 24 if is_crypto else 252 * 6.5

    # 日線
    if tf in ("1d", "d", "1day"):
        return 365 if is_crypto else 252

    # fallback: treat unknown timeframe as daily
    return 365 if is_crypto else 252


def compute(
    returns: List[float],
    equity: List[float],
    timeframe: str = "5min",
    symbols: Iterable[str] | None = None,
) -> Dict[str, Any]:
    symbols_list = list(symbols or [])
    returns_list = list(returns) if returns else [0.0]

    mu = sum(returns_list) / len(returns_list)
    sigma = stats.pstdev(returns_list) if len(returns_list) > 1 else 0.0

    af = float(_ann_factor(timeframe, symbols_list))

    ann_ret = mu * af
    ann_vol = sigma * math.sqrt(af)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    max_dd, peak = 0.0, -1e18
    for v in equity if equity else [1.0]:
        v_eff = max(v, 1e-6)
        peak = max(peak, v_eff)
        dd = (v_eff - peak) / peak
        max_dd = min(max_dd, dd)
    max_dd = max(max_dd, -1.0)

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
