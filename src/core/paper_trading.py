"""Deprecated shim for `src.core.paper_trading`."""

import warnings

warnings.warn(
    "Importing from `src.core.paper_trading` is deprecated. Use `quantproject.core.paper_trading` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.core.paper_trading import *  # noqa: F401,F403
