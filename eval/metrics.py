from __future__ import annotations

import math
import statistics as stats
from typing import Any, Dict, List


def _ann_factor(freq: str = "5min") -> float:
    # 5min: ~12 bars per hour, ~6.5 hours per day, 252 trading days
    return 252 * 6.5 * 12 if freq == "5min" else 252


def compute(returns: List[float], equity: List[float]) -> Dict[str, Any]:
    if not returns:
        returns = [0.0]
    mu = sum(returns) / len(returns)
    sigma = stats.pstdev(returns) if len(returns) > 1 else 0.0
    af = _ann_factor("5min")
    ann_ret = mu * af
    ann_vol = sigma * math.sqrt(af)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    max_dd, peak = 0.0, -1e18
    for v in (equity if equity else [1.0]):
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