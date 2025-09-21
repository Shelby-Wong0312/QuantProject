"""Deprecated shim for src.risk.risk_manager."""

import warnings

warnings.warn(
    "Importing from src.risk.risk_manager is deprecated. Use quantproject.risk.risk_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.risk_manager import *  # noqa: F401,F403
