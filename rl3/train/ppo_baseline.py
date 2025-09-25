#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

# add repo/src to sys.path so "src.quantproject..." works when running "python -m rl3.train.ppo_baseline"
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from stable_baselines3 import PPO


from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from rl3.callbacks.metrics_callback import MetricsCallback
from rl3.env.portfolio_env import EnvConfig, PortfolioEnv
from rl3.features.registry import registry
from src.quantproject.data_pipeline.loaders.bars import load_and_align


def _build_features(prices: Dict[str, pd.DataFrame], feat_names: list[str]) -> Dict[str, pd.DataFrame]:
    return {
        symbol: registry.build(df, feat_names)
        for symbol, df in prices.items()
    }


def make_env(cfg: Dict[str, Any]) -> PortfolioEnv:
    timeframe = cfg.get("timeframe", "5min")
    symbols = cfg["symbols"]
    prices = load_and_align(symbols, cfg["start"], cfg["end"], timeframe)
    feats = _build_features(prices, cfg["obs"]["features"])
    costs = cfg.get("costs", {})
    reward_cfg = cfg.get("reward", {})
    risk_cfg = cfg.get("risk", {})
    fields = EnvConfig.__dataclass_fields__
    env_cfg = EnvConfig(
        features=cfg["obs"]["features"],
        window=cfg["obs"].get("window", fields["window"].default),
        max_weight=cfg["action"].get("max_weight", fields["max_weight"].default),
        max_dweight=cfg["action"].get("max_dweight", fields["max_dweight"].default),
        gross_leverage_cap=risk_cfg.get("gross_leverage_cap", fields["gross_leverage_cap"].default),
        commission_bps=costs.get("commission_bps", fields["commission_bps"].default),
        slippage_alpha=costs.get("slippage_alpha", fields["slippage_alpha"].default),
        slippage_beta=costs.get("slippage_beta", fields["slippage_beta"].default),
        participation_cap=costs.get("participation_cap", fields["participation_cap"].default),
        reward_cost_bps=costs.get("reward_cost_bps", reward_cfg.get("cost_bps", fields["reward_cost_bps"].default)),
        lambda_turnover=reward_cfg.get("lambda_turnover", fields["lambda_turnover"].default),
        reward_clip=reward_cfg.get("clip", fields["reward_clip"].default),
        dd_hard=risk_cfg.get("dd_hard", fields["dd_hard"].default),
    )
    return PortfolioEnv(prices, feats, symbols, env_cfg)


def main(yaml_path: str) -> None:
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    log_dir = Path(cfg.get("log", {}).get("dir", "runs/rl3/baseline"))
    log_dir.mkdir(parents=True, exist_ok=True)

    env_factory = lambda: make_env(cfg)
    train_cfg = cfg["train"]
    policy_name = train_cfg.get("policy", "MlpLstmPolicy")
    policy_kwargs = train_cfg.get("policy_kwargs", {
        "net_arch": [256, 256],
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "shared_lstm": True,
    })

    env_raw = DummyVecEnv([env_factory])
    vec_norm = VecNormalize(env_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env = VecCheckNan(vec_norm, raise_exception=True)
    cb = MetricsCallback(log_dir=log_dir)

    ppo_kwargs = {
        "policy_kwargs": policy_kwargs,
        "learning_rate": train_cfg["lr"],
        "n_steps": train_cfg["n_steps"],
        "batch_size": train_cfg["batch_size"],
        "gamma": train_cfg["gamma"],
        "gae_lambda": train_cfg["gae_lambda"],
        "clip_range": train_cfg["clip_range"],
        "ent_coef": train_cfg["ent_coef"],
        "vf_coef": train_cfg["vf_coef"],
        "verbose": 1,
    }

    for optional_key in ("clip_range_vf", "max_grad_norm", "n_epochs", "seed"):
        if optional_key in train_cfg:
            ppo_kwargs[optional_key] = train_cfg[optional_key]

    model = PPO(policy_name, env, **ppo_kwargs)

    start = time.time()
    model.learn(total_timesteps=int(cfg["train"]["total_timesteps"]), callback=cb)
    elapsed = time.time() - start

    vec_norm.training = False
    vec_norm.save(str(log_dir / "vecnormalize.pkl"))
    env.close()

    model.save(log_dir / "ppo_model")

    (log_dir / "config.json").write_text(
        json.dumps(
            {
                "symbols": cfg["symbols"],
                "features": cfg["obs"]["features"],
                "timeframe": cfg.get("timeframe", "5min"),
                "total_timesteps": cfg["train"]["total_timesteps"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training done in {elapsed:.1f}s. Artifacts in {log_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)




