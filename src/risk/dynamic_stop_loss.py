"""Deprecated shim for src.risk.dynamic_stop_loss."""

import warnings

warnings.warn(
    "Importing from src.risk.dynamic_stop_loss is deprecated. Use quantproject.risk.dynamic_stop_loss instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.dynamic_stop_loss import *  # noqa: F401,F403
