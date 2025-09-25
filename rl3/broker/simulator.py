from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrokerParams:
    commission_bps: float = 1.0
    slippage_alpha: float = 0.10   # volatility-linked component (bps)
    slippage_beta: float = 0.20    # participation-linked component (bps per unit turnover)
    participation_cap: float = 0.10  # max notional per step as % of NAV


class BrokerSim:
    """Very lightweight broker cost model based on deterministic slippage assumptions."""

    def __init__(self, params: BrokerParams):
        self.p = params

    def execute(self, px: float, dweight: float, nav: float) -> float:
        """Return trade cost in base currency for the requested weight change."""
        if nav <= 0 or px <= 0 or not dweight:
            return 0.0

        notional = abs(dweight) * nav
        cap_notional = max(self.p.participation_cap, 0.0) * nav
        trade_notional = min(notional, cap_notional) if cap_notional > 0 else notional  # excess notional is deferred to future steps

        if trade_notional <= 0.0:
            return 0.0

        alpha_rate = max(self.p.slippage_alpha, 0.0) / 1e4
        beta_rate = max(self.p.slippage_beta, 0.0) / 1e4 * abs(dweight)
        impact_cost = trade_notional * (alpha_rate + beta_rate)
        commission = trade_notional * (max(self.p.commission_bps, 0.0) / 1e4)
        return commission + impact_cost
