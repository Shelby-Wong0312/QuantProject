"""Deprecated shim for `src.core`; import from :mod:`quantproject.core` instead."""

import warnings

warnings.warn(
    "Importing from `src.core` is deprecated. Please use `quantproject.core` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.core import *  # noqa: F401,F403
