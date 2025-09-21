"""Command-line helpers for PPO training workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from loguru import logger

from .configs.env_config import get_config
from .training.trainer import Trainer


PROFILE_MAP = {
    "15years": {"preset": "swing_trading", "timesteps": 2_000_000},
    "3488_stocks": {"preset": "aggressive", "timesteps": 1_500_000},
    "4000_stocks": {"preset": "aggressive", "timesteps": 1_750_000},
    "capital": {"preset": "day_trading", "timesteps": 1_200_000},
    "full": {"preset": "conservative", "timesteps": 2_500_000},
    "realistic": {"preset": "day_trading", "timesteps": 1_000_000},
    "live": {"preset": "day_trading", "timesteps": 500_000},
}


def run_training_profile(profile: str, results_dir: Optional[Path] = None) -> None:
    """Run a predefined PPO training profile using :class:`Trainer`."""
    if profile not in PROFILE_MAP:
        raise ValueError(f"Unknown PPO training profile '{profile}'.")

    profile_cfg = PROFILE_MAP[profile]
    env_config = get_config(profile_cfg["preset"])

    trainer = Trainer(
        agent_type="PPO",
        env_config=env_config,
        experiment_name=f"ppo_{profile}",
        results_dir=results_dir or Path("results")
    )
    trainer.setup_environments()
    trainer.setup_agent()
    trainer.train(total_timesteps=profile_cfg["timesteps"])


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run PPO training using QuantProject pipeline.")
    parser.add_argument("profile", choices=sorted(PROFILE_MAP.keys()))
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for training artefacts.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for Trainer (default: INFO)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger.info("Starting PPO training profile '{}'", args.profile)
    run_training_profile(args.profile, args.results_dir)
    logger.success("Training profile '{}' completed", args.profile)


if __name__ == "__main__":  # pragma: no cover
    main()
