"""Deprecated simplified PPO trainer CLI."""

import warnings

from quantproject.rl_trading.cli import main as ppo_main


if __name__ == "__main__":
    warnings.warn(
        "`simplified_ppo_trainer.py` is deprecated. Invoke `python -m quantproject.rl_trading.cli` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ppo_main()
