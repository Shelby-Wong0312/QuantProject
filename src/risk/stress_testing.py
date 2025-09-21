"""Deprecated shim for src.risk.stress_testing."""

import warnings

warnings.warn(
    "Importing from src.risk.stress_testing is deprecated. Use quantproject.risk.stress_testing instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.stress_testing import *  # noqa: F401,F403
