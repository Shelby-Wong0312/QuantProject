"""Deprecated shim; use :mod:`quantproject.rl_trading.agents.ppo_agent`."""

import warnings

from .agents.ppo_agent import *  # noqa: F401,F403

warnings.warn(
    "Importing from `quantproject.rl_trading.ppo_agent` is deprecated. Use `quantproject.rl_trading.agents.ppo_agent` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in dir() if not name.startswith('_')]
