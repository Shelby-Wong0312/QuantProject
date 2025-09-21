"""Deprecated shim for ``core``; use :mod:`quantproject.core` instead."""

from importlib import import_module
import sys
import warnings

_MESSAGE = "Importing from `core` is deprecated. Please use `quantproject.core` instead."
warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)

__all__ = ["event", "event_loop"]

for _name in __all__:
    module = import_module(f"quantproject.core.{_name}")
    sys.modules[f"core.{_name}"] = module
