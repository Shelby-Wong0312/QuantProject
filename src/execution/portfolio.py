"""Deprecated shim for `src.execution.portfolio`."""

import warnings

warnings.warn(
    "Importing from `src.execution.portfolio` is deprecated. Use `quantproject.execution.portfolio` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.execution.portfolio import *  # noqa: F401,F403
