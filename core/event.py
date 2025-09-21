"""Deprecated shim for `core.event`. Import from :mod:`quantproject.core.event`."""

import warnings

warnings.warn(
    "Importing from `core.event` is deprecated. Use `quantproject.core.event` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.core.event import *  # noqa: F401,F403
