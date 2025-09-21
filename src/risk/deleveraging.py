"""Deprecated shim for src.risk.deleveraging."""

import warnings

warnings.warn(
    "Importing from src.risk.deleveraging is deprecated. Use quantproject.risk.deleveraging instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk.deleveraging import *  # noqa: F401,F403
