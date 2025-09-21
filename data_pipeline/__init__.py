"""Deprecated shim for ``data_pipeline``; use :mod:`quantproject.data_pipeline`."""

import warnings

from quantproject.data_pipeline import *  # noqa: F401,F403

warnings.warn(
    "Importing from `data_pipeline` is deprecated. Please use `quantproject.data_pipeline` instead.",
    DeprecationWarning,
    stacklevel=2,
)
