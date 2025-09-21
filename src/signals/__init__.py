"""Deprecated shim for `src.signals`; use :mod:`quantproject.signals` instead."""

import warnings

warnings.warn(
    "Importing from `src.signals` is deprecated. Please use `quantproject.signals` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.signals import *  # noqa: F401,F403
