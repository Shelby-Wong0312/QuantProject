"""Deprecated shim; import from :mod:`quantproject.risk`."""

import warnings

from quantproject.risk import *  # noqa: F401,F403

warnings.warn(
    "Importing from `src.risk_management` is deprecated. Please import from `quantproject.risk` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RiskMetrics",
    "PositionSizing",
    "StopLoss",
    "StopType",
    "ROIMonitor",
    "ROITarget",
    "DynamicStopLoss",
    "ProfitProtection",
    "EnhancedRiskManager",
    "MarketAnomalyDetector",
    "AnomalyType",
    "SeverityLevel",
    "CircuitBreaker",
    "BreakerLevel",
    "RapidDeleveraging",
    "DeleveragingStrategy",
    "Position",
    "StressTester",
]
