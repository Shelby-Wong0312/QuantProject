#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

# add repo/src to sys.path so "src.quantproject..." works when running "python -m rl3.train.ppo_baseline"
import sys
import pathlib
import json
import time
import shutil
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from typing import Any, Dict

import pandas as pd
import yaml
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO  # type: ignore

    HAS_CONTRIB = True
except Exception:
    RecurrentPPO = None  # type: ignore
    HAS_CONTRIB = False

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from rl3.callbacks.metrics_callback import MetricsCallback
from rl3.env.portfolio_env import EnvConfig, PortfolioEnv
from rl3.env.data_roots import ensure_data_roots
from rl3.features.registry import registry
from src.quantproject.data_pipeline.loaders.bars import load_and_align


def _build_features(
    prices: Dict[str, pd.DataFrame], feat_names: list[str]
) -> Dict[str, pd.DataFrame]:
    return {symbol: registry.build(df, feat_names) for symbol, df in prices.items()}


def make_env(cfg: Dict[str, Any]) -> PortfolioEnv:
    ensure_data_roots(cfg)
    timeframe = cfg.get("timeframe", "5min")
    cfg["symbols"]
    # === inserted: fetch symbols/timeframe from cfg ===
    symbols = (
        cfg.get('symbols')
        or cfg.get('env', {}).get('symbols')
        or cfg.get('dataset', {}).get('symbols')
    )
    timeframe = (
        cfg.get('timeframe')
        or cfg.get('dataset', {}).get('freq')
        or '5m'
    )
    if not symbols:
        raise ValueError('symbols not found in cfg (checked cfg.symbols / env.symbols / dataset.symbols)')
    # === end inserted ===
    prices = load_and_align((cfg.get("symbols") or cfg.get("env", {}).get("symbols") or cfg.get("dataset", {}).get("symbols")), cfg["start"], cfg["end"], (cfg.get("timeframe") or cfg.get("dataset", {}).get("freq") or "5m"))
    fields = EnvConfig.__dataclass_fields__
    obs_cfg = cfg["obs"]
    window = int(obs_cfg.get("window", fields["window"].default))
    required_bars = window + 1
    min_len = min(len(df) for df in prices.values())
    if min_len < required_bars:
        raise ValueError(
            f"Not enough bars for window={window}: got {min_len}. Require >= {required_bars}."
        )
    feats = _build_features(prices, obs_cfg["features"])
    costs = cfg.get("costs", {})
    reward_cfg = cfg.get("reward", {})
    risk_cfg = cfg.get("risk", {})
    act = cfg.get("action", {})
    dwt = act.get("dweight_threshold")
    if dwt is None:
        dwt = cfg.get("dweight_threshold")
    thr = dwt if dwt is not None else fields["dweight_threshold"].default
    env_cfg = EnvConfig(
        features=obs_cfg["features"],
        window=window,
        max_weight=act.get("max_weight", fields["max_weight"].default),
        max_dweight=act.get("max_dweight", fields["max_dweight"].default),
        dweight_threshold=float(thr),
        gross_leverage_cap=risk_cfg.get(
            "gross_leverage_cap", fields["gross_leverage_cap"].default
        ),
        commission_bps=costs.get("commission_bps", fields["commission_bps"].default),
        slippage_alpha=costs.get("slippage_alpha", fields["slippage_alpha"].default),
        slippage_beta=costs.get("slippage_beta", fields["slippage_beta"].default),
        participation_cap=costs.get(
            "participation_cap", fields["participation_cap"].default
        ),
        reward_cost_bps=costs.get(
            "reward_cost_bps",
            reward_cfg.get("cost_bps", fields["reward_cost_bps"].default),
        ),
        lambda_turnover=reward_cfg.get(
            "lambda_turnover", fields["lambda_turnover"].default
        ),
        reward_clip=reward_cfg.get("clip", fields["reward_clip"].default),
        dd_hard=risk_cfg.get("dd_hard", fields["dd_hard"].default),
    )
    return PortfolioEnv(prices, feats, symbols, env_cfg)


def main(yaml_path: str) -> None:
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    log_dir = Path(cfg.get("log", {}).get("dir", "runs/rl3/baseline"))
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["train"]
    policy_name = train_cfg.get("policy", "MlpPolicy")
    policy_kwargs = train_cfg.get(
        "policy_kwargs",
        {"net_arch": [256, 256]},
    )

    algo_cls = PPO
    if policy_name == "MlpLstmPolicy":
        if HAS_CONTRIB and RecurrentPPO is not None:
            algo_cls = RecurrentPPO
        else:
            print("[WARN] sb3-contrib not installed; falling back to PPO + MlpPolicy")
            policy_name = "MlpPolicy"
            for key in (
                "lstm_hidden_size",
                "n_lstm_layers",
                "shared_lstm",
                "enable_critic_lstm",
            ):
                policy_kwargs.pop(key, None)

    config_payload = json.dumps(
        {
            "symbols": cfg["symbols"],
            "features": cfg["obs"]["features"],
            "timeframe": cfg.get("timeframe", "5min"),
            "window": cfg["obs"].get(
                "window", EnvConfig.__dataclass_fields__["window"].default
            ),
            "action": cfg.get("action", {}),
            "obs": cfg.get("obs", {}),
            "risk": cfg.get("risk", {}),
            "costs": cfg.get("costs", {}),
            "train": cfg.get("train", {}),
            "total_timesteps": train_cfg["total_timesteps"],
        },
        indent=2,
    )
    (log_dir / "config.json").write_text(config_payload, encoding="utf-8")

    def env_factory():
        return make_env(cfg)

    env_raw = DummyVecEnv([env_factory])
    vec_norm = VecNormalize(env_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env = VecCheckNan(vec_norm, raise_exception=True)
    cb = MetricsCallback(log_dir=log_dir)

    extra_kwargs: Dict[str, Any] = {}
    for optional_key in ("clip_range_v", "max_grad_norm", "n_epochs", "seed"):
        if optional_key in train_cfg:
            extra_kwargs[optional_key] = train_cfg[optional_key]

    model = algo_cls(
        policy_name,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=train_cfg["lr"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg["ent_coef"],
        vf_coef=train_cfg["vf_coef"],
        verbose=1,
        **extra_kwargs,
    )

    start = time.time()
    model.learn(total_timesteps=int(train_cfg["total_timesteps"]), callback=cb)
    elapsed = time.time() - start

    vec_norm.training = False
    vec_norm.norm_reward = False
    model_path = log_dir / "model.zip"
    vecnorm_path = log_dir / "vecnormalize.pkl"

    log_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    try:
        vec_norm.save(str(vecnorm_path))
    except Exception:
        pass
    env.close()

    legacy_path = log_dir / "ppo_model.zip"
    try:
        shutil.copyfile(model_path, legacy_path)
    except Exception:
        pass

    print(f"Training done in {elapsed:.1f}s. Artifacts in {log_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
