"""Deprecated shim for src.risk.circuit_breaker."""

import warnings

warnings.warn(
    "Importing from src.risk.circuit_breaker is deprecated. Use quantproject.risk.circuit_breaker instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.circuit_breaker import *  # noqa: F401,F403
