"""Deprecated shim for src.risk_management.risk_metrics."""

import warnings

warnings.warn(
    "Importing from src.risk_management.risk_metrics is deprecated. Use quantproject.risk.risk_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.risk_metrics import *  # noqa: F401,F403
