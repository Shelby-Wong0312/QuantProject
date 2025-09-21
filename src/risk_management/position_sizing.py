"""Deprecated shim for src.risk_management.position_sizing."""

import warnings

warnings.warn(
    "Importing from src.risk_management.position_sizing is deprecated. Use quantproject.risk.position_sizing instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.position_sizing import *  # noqa: F401,F403
