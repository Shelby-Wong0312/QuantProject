"""Deprecated shim for `src.data_pipeline.data_manager`."""

import warnings

warnings.warn(
    "Importing from `src.data_pipeline.data_manager` is deprecated. Use `quantproject.data_pipeline.data_manager` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.data_pipeline.data_manager import *  # noqa: F401,F403
