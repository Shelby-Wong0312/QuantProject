"""Deprecated shim for `src.execution`; import from :mod:`quantproject.execution` instead."""

import warnings

warnings.warn(
    "Importing from `src.execution` is deprecated. Please use `quantproject.execution` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.execution import *  # noqa: F401,F403
