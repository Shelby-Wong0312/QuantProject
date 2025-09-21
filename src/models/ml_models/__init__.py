"""Deprecated shim for `src.models.ml_models`; import from :mod:`quantproject.models.ml_models`."""

import warnings

warnings.warn(
    "Importing from `src.models.ml_models` is deprecated. Please use `quantproject.models.ml_models` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quantproject.models.ml_models import *  # noqa: F401,F403
