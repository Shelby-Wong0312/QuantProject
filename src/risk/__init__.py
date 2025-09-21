"""Deprecated shim for `src.risk`; use :mod:`quantproject.risk`.""" 

import warnings

warnings.warn(
    "Importing from `src.risk` is deprecated. Please use `quantproject.risk` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.risk import *  # noqa: F401,F403
