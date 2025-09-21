"""Deprecated PPO live utility wrapper."""

import warnings

from quantproject.rl_trading.cli import run_training_profile


def _run():
    warnings.warn(
        "This script is deprecated. Use `python -m quantproject.rl_trading.cli live` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    run_training_profile("live")


if __name__ == "__main__":
    _run()
