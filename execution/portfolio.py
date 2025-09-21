"""Deprecated shim for `execution.portfolio`. Import from :mod:`quantproject.execution.portfolio`."""

import warnings

warnings.warn(
    "Importing from `execution.portfolio` is deprecated. Use `quantproject.execution.portfolio` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.execution.portfolio import *  # noqa: F401,F403
