"""Deprecated shim for src.risk.anomaly_detection."""

import warnings

warnings.warn(
    "Importing from src.risk.anomaly_detection is deprecated. Use quantproject.risk.anomaly_detection instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.anomaly_detection import *  # noqa: F401,F403
