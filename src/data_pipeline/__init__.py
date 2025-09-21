"""Deprecated shim for `src.data_pipeline`; use :mod:`quantproject.data_pipeline`."""

import warnings

warnings.warn(
    "Importing from `src.data_pipeline` is deprecated. Please use `quantproject.data_pipeline` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.data_pipeline import *  # noqa: F401,F403
