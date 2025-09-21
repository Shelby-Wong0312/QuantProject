"""Deprecated shim for src.risk.roi_monitor."""

import warnings

warnings.warn(
    "Importing from src.risk.roi_monitor is deprecated. Use quantproject.risk.roi_monitor instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.roi_monitor import *  # noqa: F401,F403
