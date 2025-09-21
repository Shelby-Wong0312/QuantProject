"""Deprecated shim for `src.core.event_loop`."""

import warnings

warnings.warn(
    "Importing from `src.core.event_loop` is deprecated. Use `quantproject.core.event_loop` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.core.event_loop import *  # noqa: F401,F403
