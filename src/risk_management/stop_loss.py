"""Deprecated shim for src.risk_management.stop_loss."""

import warnings

warnings.warn(
    "Importing from src.risk_management.stop_loss is deprecated. Use quantproject.risk.stop_loss instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.stop_loss import *  # noqa: F401,F403
