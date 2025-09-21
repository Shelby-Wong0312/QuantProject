"""Deprecated shim for `src.core.event`."""

import warnings

warnings.warn(
    "Importing from `src.core.event` is deprecated. Use `quantproject.core.event` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.core.event import *  # noqa: F401,F403
