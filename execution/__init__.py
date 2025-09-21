"""Deprecated shim for ``execution``; use :mod:`quantproject.execution` instead."""

from importlib import import_module
import sys
import warnings

_MESSAGE = "Importing from `execution` is deprecated. Please use `quantproject.execution` instead."
warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)

__all__ = ["broker", "portfolio"]

for _name in __all__:
    module = import_module(f"quantproject.execution.{_name}")
    sys.modules[f"execution.{_name}"] = module
