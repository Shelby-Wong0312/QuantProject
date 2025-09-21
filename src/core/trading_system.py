"""Deprecated shim for `src.core.trading_system`."""

import warnings

warnings.warn(
    "Importing from `src.core.trading_system` is deprecated. Use `quantproject.core.trading_system` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# removed invalid self-import shim
