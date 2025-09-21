"""Deprecated shim for `src.data_pipeline.live_feed`."""

import warnings

warnings.warn(
    "Importing from `src.data_pipeline.live_feed` is deprecated. Use `quantproject.data_pipeline.live_feed` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.data_pipeline.live_feed import *  # noqa: F401,F403
