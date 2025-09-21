"""Deprecated shim for `src.signals.signal_generator`."""

import warnings

warnings.warn(
    "Importing from `src.signals.signal_generator` is deprecated. Use `quantproject.signals.signal_generator` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.signals.signal_generator import *  # noqa: F401,F403
