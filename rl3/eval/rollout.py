#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys, pathlib
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import json as _json
import pandas as pd
import yaml
from stable_baselines3 import PPO
import torch

try:
    from sb3_contrib import RecurrentPPO  # type: ignore

    HAS_CONTRIB = True
except Exception:
    RecurrentPPO = None  # type: ignore
    HAS_CONTRIB = False

from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize

from eval.metrics import compute
from eval.reports import write_markdown
from rl3.env.portfolio_env import EnvConfig, PortfolioEnv


def _timeframe_minutes(timeframe: Optional[str]) -> Optional[int]:
    if timeframe is None:
        return None
    tf = str(timeframe).strip().lower()
    if not tf:
        return None
    if tf in {"1d", "day", "daily"}:
        return 1440
    for suffix, multiplier in (("min", 1), ("m", 1), ("h", 60)):
        if tf.endswith(suffix):
            token = tf[: -len(suffix)] or "1"
            try:
                return int(float(token) * multiplier)
            except ValueError:
                return None
    try:
        return int(float(tf))
    except ValueError:
        return None


from rl3.features.registry import registry
from src.quantproject.data_pipeline.loaders.bars import load_and_align_router


def _write_oos_meta(
    out_dir: Path,
    cfg: Dict[str, Any],
    model_path: Path,
    vecnorm_path: Path,
    start: str,
    end: str,
    deterministic: bool,
) -> None:
    payload = {
        "config": str(cfg.get("__cfg_path__", "")) or str(cfg.get("config", "")),
        "model": str(model_path),
        "vecnormalize": str(vecnorm_path),
        "start": start,
        "end": end,
        "timeframe": cfg.get("timeframe", "5min"),
        "symbols": list(cfg.get("symbols", [])),
        "deterministic": bool(deterministic),
    }
    (out_dir / "oos_meta.json").write_text(_json.dumps(payload, indent=2), encoding="utf-8")


def _build_features(
    prices: Dict[str, pd.DataFrame], feat_names: List[str]
) -> Dict[str, pd.DataFrame]:
    return {symbol: registry.build(df, feat_names) for symbol, df in prices.items()}


def _make_env(cfg: Dict[str, Any], start: str, end: str) -> PortfolioEnv:
    timeframe = cfg.get("timeframe", "5min")
    symbols = list(cfg.get("symbols", []))
    if not symbols:
        raise ValueError("Config must specify symbols for evaluation.")

    obs_cfg = cfg.get("obs") or {}
    fields = EnvConfig.__dataclass_fields__
    window = int(obs_cfg.get("window", cfg.get("window", fields["window"].default)))

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    lookback_days = max(3, window // 6 + 1)
    max_lookback_days = 120
    prices = None

    while True:
        prefetch_ts = start_ts - pd.Timedelta(days=lookback_days)
        prefetch_start = prefetch_ts.isoformat()
        prices = load_and_align_router(symbols, prefetch_start, end, timeframe)
        if not prices:
            lookback_days += 3
            if lookback_days > max_lookback_days:
                raise ValueError("No price data returned for the requested window.")
            continue
        min_len = min(len(df) for df in prices.values())
        if min_len > window:
            break
        lookback_days += 3
        if lookback_days > max_lookback_days:
            raise ValueError("Insufficient history to build observations for evaluation.")

    feat_names = list(obs_cfg.get("features") or cfg.get("features", []))
    if not feat_names:
        raise ValueError("Config must provide features for evaluation.")
    feats = _build_features(prices, feat_names)
    required_bars = window + 1
    min_len = min(len(df) for df in prices.values())
    if min_len < required_bars:
        raise ValueError(
            f"Not enough bars for window={window}: got {min_len}. Require >= {required_bars}."
        )

    action_cfg = cfg.get("action", {})
    risk_cfg = cfg.get("risk", {})
    cost_cfg = cfg.get("costs", {})
    reward_cfg = cfg.get("reward", {})

    fields = EnvConfig.__dataclass_fields__
    thr = action_cfg.get("dweight_threshold")
    if thr is None:
        thr = cfg.get("dweight_threshold")
    if thr is None:
        thr = fields["dweight_threshold"].default
    thr = float(thr)
    env_cfg = EnvConfig(
        features=feat_names,
        window=obs_cfg.get("window", cfg.get("window", fields["window"].default)),
        max_weight=action_cfg.get(
            "max_weight", cfg.get("max_weight", fields["max_weight"].default)
        ),
        max_dweight=action_cfg.get(
            "max_dweight", cfg.get("max_dweight", fields["max_dweight"].default)
        ),
        dweight_threshold=thr,
        gross_leverage_cap=risk_cfg.get(
            "gross_leverage_cap",
            cfg.get("gross_leverage_cap", fields["gross_leverage_cap"].default),
        ),
        commission_bps=cost_cfg.get(
            "commission_bps", cfg.get("commission_bps", fields["commission_bps"].default)
        ),
        slippage_alpha=cost_cfg.get(
            "slippage_alpha", cfg.get("slippage_alpha", fields["slippage_alpha"].default)
        ),
        slippage_beta=cost_cfg.get(
            "slippage_beta", cfg.get("slippage_beta", fields["slippage_beta"].default)
        ),
        participation_cap=cost_cfg.get(
            "participation_cap", cfg.get("participation_cap", fields["participation_cap"].default)
        ),
        reward_cost_bps=cost_cfg.get(
            "reward_cost_bps",
            cfg.get(
                "reward_cost_bps", reward_cfg.get("cost_bps", fields["reward_cost_bps"].default)
            ),
        ),
        lambda_turnover=reward_cfg.get(
            "lambda_turnover", cfg.get("lambda_turnover", fields["lambda_turnover"].default)
        ),
        reward_clip=reward_cfg.get("clip", cfg.get("reward_clip", fields["reward_clip"].default)),
        dd_hard=risk_cfg.get("dd_hard", cfg.get("dd_hard", fields["dd_hard"].default)),
    )
    return PortfolioEnv(prices, feats, symbols, env_cfg)


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise TypeError("Config root must be a mapping.")
    cfg.setdefault("__cfg_path__", str(cfg_path))
    return cfg


def _load_model_any(model_path: Path, env):
    if HAS_CONTRIB and RecurrentPPO is not None:
        try:
            return RecurrentPPO.load(str(model_path), env=env, print_system_info=False)
        except Exception:
            pass
    return PPO.load(str(model_path), env=env, print_system_info=False)


def _compute_trade_stats(
    weights: np.ndarray,
    threshold: float,
    timeframe: Optional[str],
    start: str,
    end: str,
) -> Dict[str, float]:
    if weights.size == 0 or weights.shape[0] < 2:
        return {
            "step_hits": 0.0,
            "accum_hits": 0.0,
            "trades_week_step": 0.0,
            "trades_week_accum": 0.0,
        }
    if weights.ndim == 1:
        weights = weights[:, np.newaxis]
    diffs = np.abs(np.diff(weights, axis=0))
    diffs = np.nan_to_num(diffs, nan=0.0)
    step_hits = float((diffs >= threshold).any(axis=1).sum())
    accum_vec = np.zeros(diffs.shape[1], dtype=float)
    accum_hits = 0.0
    for row in diffs:
        accum_vec += row
        mask = accum_vec >= threshold
        if mask.any():
            accum_hits += 1.0
            accum_vec[mask] = 0.0

    tf_minutes = _timeframe_minutes(timeframe)
    bars_per_week = (7 * 24 * 60) / tf_minutes if tf_minutes and tf_minutes > 0 else 168.0
    bars_weeks = max(1e-9, weights.shape[0] / bars_per_week) if bars_per_week > 0 else 1e-9
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        seconds = max(0.0, (end_ts - start_ts).total_seconds())
        meta_weeks = max(1e-9, seconds / (7 * 24 * 3600.0))
    except Exception:
        meta_weeks = 0.0
    weeks = max(meta_weeks, bars_weeks, 1e-9)
    return {
        "step_hits": float(step_hits),
        "accum_hits": float(accum_hits),
        "trades_week_step": float(step_hits / weeks),
        "trades_week_accum": float(accum_hits / weeks),
    }


def run_oos_eval(
    model_path: Path,
    vecnorm_path: Path,
    cfg_path: Path,
    start: str,
    end: str,
    out_dir: Path,
    deterministic: bool = True,
) -> Dict[str, float]:
    cfg = _load_config(cfg_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found. Expected at: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found. Expected at: {vecnorm_path}")

    env_factory = lambda: _make_env(cfg, start, end)
    base_env = DummyVecEnv([env_factory])

    try:
        env = VecNormalize.load(str(vecnorm_path), base_env)
    except Exception as exc:
        raise RuntimeError(f"Failed to load VecNormalize stats from {vecnorm_path}: {exc}") from exc

    env.training = False
    env.norm_reward = False
    env = VecCheckNan(env, raise_exception=True)

    try:
        model = _load_model_any(model_path, env)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {model_path}: {exc}") from exc

    model.policy.set_training_mode(False)

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out

    policy = getattr(model, "policy", None)
    is_recurrent = bool(getattr(policy, "recurrent", False) or hasattr(policy, "lstm_actor"))
    num_envs = getattr(env, "num_envs", 1)
    if num_envs != 1:
        raise RuntimeError(
            "OOS rollout currently assumes num_envs == 1; please wrap with DummyVecEnv(1)."
        )

    lstm_states: Optional[Tuple[np.ndarray, ...]] = None
    assert num_envs == 1, "OOS rollout only supports a single environment instance."
    episode_starts = np.array([True], dtype=bool)
    episode_resets = 1

    if is_recurrent and hasattr(policy, "initial_state"):
        try:
            lstm_states = policy.initial_state(num_envs)
        except Exception:
            lstm_states = None

    dones = np.array([False], dtype=bool)
    returns: List[float] = []
    equity: List[float] = []
    weights: List[List[float]] = []

    done = False

    with torch.no_grad():
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info_payload = step_out
                terminated = bool(np.asarray(terminated).item())
                truncated = bool(np.asarray(truncated).item())
                done = terminated or truncated
                if is_recurrent:
                    episode_starts = np.array([done], dtype=bool)
            else:
                obs, reward, done_arr, info_payload = step_out
                done = bool(np.asarray(done_arr).item())
                if is_recurrent:
                    episode_starts = np.array([done], dtype=bool)

            info = (
                info_payload[0] if isinstance(info_payload, (list, tuple)) else info_payload or {}
            )
            reward_val = (
                float(np.asarray(reward).item()) if np.asarray(reward).size else float(reward)
            )
            pnl_ret = float(info.get("pnl_ret", reward_val))
            cost_rate = float(info.get("cost_rate", 0.0))
            returns.append(pnl_ret - cost_rate)
            equity.append(float(info.get("nav", 0.0)))
            weight_vec = info.get("weights")
            if weight_vec is not None:
                weights.append([float(x) for x in weight_vec])

    weights_arr = np.asarray(weights, dtype=float) if weights else np.empty((0, 0), dtype=float)
    if weights_arr.size == 0:
        unique_weights = 0
        total_dweight = 0.0
    else:
        rounded = np.round(weights_arr, 10)
        unique_weights = int(np.unique(rounded, axis=0).shape[0])
        total_dweight = (
            float(np.abs(np.diff(weights_arr, axis=0)).sum()) if len(weights_arr) > 1 else 0.0
        )

    metrics = compute(
        returns=returns,
        equity=equity,
        timeframe=cfg.get("timeframe", "5min"),
        symbols=cfg.get("symbols", []),
    )
    metrics["unique_weights"] = int(unique_weights)
    metrics["total_dweight"] = float(total_dweight)
    metrics["episode_resets"] = int(episode_resets)

    action_cfg = cfg.get("action", {})
    fields = EnvConfig.__dataclass_fields__
    thr = action_cfg.get("dweight_threshold")
    if thr is None:
        thr = cfg.get("dweight_threshold")
    if thr is None:
        thr = fields["dweight_threshold"].default
    thr = float(thr)
    trade_stats = _compute_trade_stats(weights_arr, thr, cfg.get("timeframe"), start, end)
    metrics.update(trade_stats)
    metrics["dweight_threshold"] = float(thr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oos_returns.json").write_text(json.dumps(returns), encoding="utf-8")
    (out_dir / "oos_equity.json").write_text(json.dumps(equity), encoding="utf-8")
    (out_dir / "oos_weights.json").write_text(json.dumps(weights), encoding="utf-8")
    if weights:
        pd.DataFrame(weights).to_csv(out_dir / "weights.csv")
    (out_dir / "oos_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _write_oos_meta(out_dir, cfg, model_path, vecnorm_path, start, end, deterministic)

    report_lines = [
        "# RL3  OOS Report",
        "## Summary",
        f"- Annualized Return: {metrics['ann_return']:.4f}",
        f"- Annualized Vol: {metrics['ann_vol']:.4f}",
        f"- Sharpe: {metrics['sharpe']:.3f}",
        f"- Max Drawdown: {metrics['max_drawdown']:.3%}",
        f"- Symbols: {', '.join(cfg.get('symbols', []))}",
        f"- Period: {start}  {end}",
    ]
    (out_dir / "oos_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return metrics


__all__ = ["run_oos_eval"]
