"""Deprecated CLI wrapper for PPO training profile 'capital'."""

import warnings

from quantproject.rl_trading.cli import run_training_profile


if __name__ == "__main__":
    warnings.warn(
        "This script is deprecated. Use `python -m quantproject.rl_trading.cli capital` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    run_training_profile("capital")
