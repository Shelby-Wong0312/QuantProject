"""Quick entry point for PPO training (deprecated)."""

import argparse
import warnings

from quantproject.rl_trading.cli import run_training_profile


if __name__ == "__main__":
    warnings.warn(
        "`START_PPO_TRAINING_NOW.py` is deprecated. Use `python -m quantproject.rl_trading.cli` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="Run a PPO training profile immediately.")
    parser.add_argument("profile", choices=[
        "15years",
        "3488_stocks",
        "4000_stocks",
        "capital",
        "full",
        "realistic",
        "live",
    ], default="realistic")
    args = parser.parse_args()
    run_training_profile(args.profile)
