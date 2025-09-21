"""Deprecated entry point for PPO trader training."""

import warnings

from quantproject.rl_trading.cli import main as ppo_main


if __name__ == "__main__":
    warnings.warn(
        "`scripts/train_ppo_trader.py` is deprecated. Use `python -m quantproject.rl_trading.cli` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ppo_main()
