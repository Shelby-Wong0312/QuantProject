"""Deprecated shim for src.risk.risk_manager_enhanced."""

import warnings

warnings.warn(
    "Importing from src.risk.risk_manager_enhanced is deprecated. Use quantproject.risk.risk_manager_enhanced instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.risk_manager_enhanced import *  # noqa: F401,F403
