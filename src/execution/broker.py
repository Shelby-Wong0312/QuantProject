"""Deprecated shim for `src.execution.broker`."""

import warnings

warnings.warn(
    "Importing from `src.execution.broker` is deprecated. Use `quantproject.execution.broker` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.execution.broker import *  # noqa: F401,F403
