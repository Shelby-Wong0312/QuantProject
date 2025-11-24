#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import random
import sys, pathlib
from pathlib import Path
import os

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

try:
    from rl3.eval.vecenv_compat import _CompatExposeSingleSpace
except ImportError:  # pragma: no cover
    from quant_project_RL.rl3.eval.vecenv_compat import _CompatExposeSingleSpace  # type: ignore

from rl3.eval.metrics import compute
from rl3.env.portfolio_env import EnvConfig, PortfolioEnv
from rl3.env.data_roots import ensure_data_roots

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
    model_path: Path | None,
    vecnorm_path: Path | None,
    start: str,
    end: str,
    deterministic: bool,
    calendar: str | None,
    seed: int | None,
    *,
    bars: int,
    bars_per_week: float | None,
    dweight_threshold: float,
    max_dweight: float,
    lambda_turnover: float,
    random_policy: bool,
) -> None:
    payload = {
        "config": str(cfg.get("__cfg_path__", "")) or str(cfg.get("config", "")),
        "model": str(model_path) if model_path is not None else None,
        "vecnormalize": str(vecnorm_path) if vecnorm_path is not None else None,
        "start": start,
        "end": end,
        "timeframe": cfg.get("timeframe", "5min"),
        "symbols": list(cfg.get("symbols", [])),
        "deterministic": bool(deterministic),
        "calendar": calendar,
        "seed": seed,
        "bars": int(bars),
        "bars_per_week": float(bars_per_week) if bars_per_week is not None else None,
        "dweight_threshold": float(dweight_threshold),
        "max_dweight": float(max_dweight),
        "lambda_turnover": float(lambda_turnover),
        "policy": "random" if random_policy else "model",
    }
    (out_dir / "oos_meta.json").write_text(_json.dumps(payload, indent=2), encoding="utf-8")


def _build_features(
    prices: Dict[str, pd.DataFrame], feat_names: List[str]
) -> Dict[str, pd.DataFrame]:
    return {symbol: registry.build(df, feat_names) for symbol, df in prices.items()}


def _sanitize_symbol(value: str) -> str:
    token = value.strip()
    for ch in (" ", "/", "\\", ":", "|", ","):
        token = token.replace(ch, "_")
    return token or "asset"


def _make_env(
    cfg: Dict[str, Any],
    start: str,
    end: str,
    data_router: Any | None = None,
) -> PortfolioEnv:
    ensure_data_roots(cfg)
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
        prices = load_and_align_router(
            symbols,
            prefetch_start,
            end,
            timeframe,
            router=data_router,
        )
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

    if not prices:
        raise ValueError("Failed to retrieve price data for evaluation.")

    feat_names = list(obs_cfg.get("features") or cfg.get("features", []))
    if not feat_names:
        cfg["features"] = ["close"]
        feat_names = ["close"]
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


def _unwrap_env(env: Any) -> Any:
    node = env
    visited: set[int] = set()
    while hasattr(node, "venv") and id(node) not in visited:
        visited.add(id(node))
        node = node.venv
    if hasattr(node, "envs") and getattr(node, "envs", None):
        node = node.envs[0]
    while hasattr(node, "env"):
        node = node.env
    return node


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
    model_path: Path | str | None,
    vecnorm_path: Path | str | None,
    cfg_path: Path | str | None,
    start: str,
    end: str,
    out_dir: Path | str,
    deterministic: bool = True,
    *,
    cfg_override: Dict[str, Any] | None = None,
    data_router: Any | None = None,
    calendar: str | None = None,
    seed: int | None = None,
    random_policy: bool = False,
) -> Dict[str, float]:
    model_path = Path(model_path).expanduser() if model_path is not None else None
    if vecnorm_path is not None:
        vecnorm_path = Path(vecnorm_path).expanduser()
    out_dir = Path(out_dir).expanduser()
    cfg_path_resolved = Path(cfg_path).expanduser() if cfg_path is not None else None

    if cfg_override is not None:
        cfg = copy.deepcopy(cfg_override)
        if cfg_path_resolved is not None:
            cfg.setdefault("__cfg_path__", str(cfg_path_resolved))
    else:
        if cfg_path_resolved is None:
            raise ValueError("cfg_path is required when cfg_override is not provided.")
        cfg = _load_config(cfg_path_resolved)

    if random_policy:
        if model_path is not None:
            raise ValueError("model_path must be omitted when using random_policy.")
    else:
        if model_path is None:
            raise ValueError("model_path is required unless random_policy is enabled.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found. Expected at: {model_path}")

    if vecnorm_path is not None and not vecnorm_path.exists():
        print(
            f"Warning: VecNormalize file not found at {vecnorm_path}, proceeding without normalization."
        )
        vecnorm_path = None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(int(seed))
        except Exception:
            pass

    env_factory = lambda: _make_env(cfg, start, end, data_router)
    base_env = DummyVecEnv([env_factory])
    # === OOS progress hook ===
    try:
        _oos_progress_counter = {'n': 0}
        _oos_report_every = 1000  # ? 1000 steps ?????????
    
        _oos_orig_step = env.step
    
        def _oos_step_with_progress(action, _orig=_oos_orig_step, _env=env, _cnt=_oos_progress_counter, _every=_oos_report_every):
            _cnt['n'] += 1
            if _cnt['n'] % _every == 0:
                total = getattr(_env, 'max_steps', None) or getattr(_env, 'n_steps', None)
                if total:
                    pct = 100.0 * _cnt['n'] / float(total)
                    print('[OOS] steps {}/{} ({:.1f}%)'.format(_cnt['n'], total, pct), flush=True)
                else:
                    print('[OOS] steps {}'.format(_cnt['n']), flush=True)
            return _orig(action)
    
        env.step = _oos_step_with_progress
    except Exception as _e:
        print('[OOS] progress hook install failed: {}'.format(_e), flush=True)
    # =========================
    if seed is not None:
        try:
            base_env.seed(seed)
        except Exception:
            pass

    used_vecnormalize = False
    if vecnorm_path is not None:
        try:
            env = VecNormalize.load(str(vecnorm_path), base_env)
            used_vecnormalize = True
        except Exception as exc:
            raise RuntimeError(f"Failed to load VecNormalize stats from {vecnorm_path}: {exc}") from exc
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    env = VecCheckNan(env, raise_exception=True)
    env = _CompatExposeSingleSpace(env)

    model = None
    if not random_policy:
        try:
            model = _load_model_any(model_path, env)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model {model_path}: {exc}") from exc
        model.policy.set_training_mode(False)

    try:
        reset_out = env.reset(seed=seed)
    except TypeError:
        reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out
    obs = np.asarray(obs)

    policy = getattr(model, "policy", None)
    is_recurrent = bool(getattr(policy, "recurrent", False) or hasattr(policy, "lstm_actor"))
    num_envs = getattr(env, "num_envs", 1)
    single_space = getattr(env, "single_action_space", None)
    if single_space is None or getattr(single_space, "shape", None) is None:
        raise RuntimeError("VecEnv must expose single_action_space with a defined shape.")
    single_action_shape = tuple(single_space.shape)
    if num_envs != 1:
        raise RuntimeError(
            "OOS rollout currently assumes num_envs == 1; please wrap with DummyVecEnv(1)."
        )

    lstm_states: Optional[Tuple[np.ndarray, ...]] = None
    assert num_envs == 1, "OOS rollout only supports a single environment instance."
    episode_starts = np.array([True], dtype=bool)
    episode_resets = 1

    if not random_policy and is_recurrent and hasattr(policy, "initial_state"):
        try:
            lstm_states = policy.initial_state(num_envs)
        except Exception:
            lstm_states = None

    returns: List[float] = []
    step_weights: List[List[float]] = []

    done = False

    with torch.no_grad():
        while not done:
            if random_policy:
                # 構造 batched 動作 (num_envs, *single_shape)
                action = np.zeros((num_envs,) + single_action_shape, dtype=np.float32)  # 全零Δ
                step_out = env.step(action)
            else:
                if is_recurrent:
                    act, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=deterministic,
                    )
                else:
                    act, _ = model.predict(obs, deterministic=deterministic)
                act_arr = np.asarray(act, dtype=np.float32)
                if act_arr.shape == single_action_shape:
                    act_arr = act_arr.reshape((num_envs,) + single_action_shape)
                elif act_arr.shape != (num_envs,) + single_action_shape:
                    raise ValueError(
                        f"Model produced action of shape {act_arr.shape}, expected {(num_envs,) + single_action_shape}"
                    )
                step_out = env.step(act_arr)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info_payload = step_out
                terminated = bool(np.asarray(terminated).item())
                truncated = bool(np.asarray(truncated).item())
                done = terminated or truncated
                if not random_policy and is_recurrent:
                    episode_starts = np.array([done], dtype=bool)
            else:
                obs, reward, done_arr, info_payload = step_out
                done = bool(np.asarray(done_arr).item())
                if not random_policy and is_recurrent:
                    episode_starts = np.array([done], dtype=bool)

            obs = np.asarray(obs)

            info = (
                info_payload[0] if isinstance(info_payload, (list, tuple)) else info_payload or {}
            )
            reward_val = (
                float(np.asarray(reward).item()) if np.asarray(reward).size else float(reward)
            )
            pnl_ret = float(info.get("pnl_ret", reward_val))
            cost_rate = float(info.get("cost_rate", 0.0))
            returns.append(pnl_ret - cost_rate)
            weight_vec = info.get("weights")
            if weight_vec is not None:
                step_weights.append([float(x) for x in weight_vec])

    base_env_unwrapped = _unwrap_env(env)
    initial_nav = float(getattr(base_env_unwrapped, "initial_nav", 1.0))

    traj = None
    if hasattr(base_env_unwrapped, "get_episode_trajectory"):
        try:
            traj = base_env_unwrapped.get_episode_trajectory()
        except Exception:
            traj = None

    if traj and traj.get("returns") is not None:
        returns = [float(x) for x in traj.get("returns", returns)]

    if traj and traj.get("equity"):
        equity_history = [float(x) for x in traj.get("equity", [])]
    else:
        equity_history = [initial_nav]
        for r in returns:
            equity_history.append(equity_history[-1] * (1.0 + r))

    returns_count = len(returns)
    if returns_count == 0:
        raise RuntimeError(
            "Rollout produced zero evaluation steps; ensure the window covers at least one bar."
        )

    raw_weights_history = None
    if traj and traj.get("weights"):
        raw_weights_history = [[float(v) for v in row] for row in traj.get("weights", [])]

    asset_dim = len(step_weights[0]) if step_weights else int(getattr(base_env_unwrapped, "N", 0))
    if raw_weights_history:
        weights_arr_full = np.asarray(raw_weights_history, dtype=float)
    else:
        step_weights_array_tmp = np.asarray(step_weights, dtype=float)
        if step_weights_array_tmp.ndim == 1 and asset_dim:
            step_weights_array_tmp = step_weights_array_tmp.reshape(-1, asset_dim)
        if step_weights_array_tmp.size == 0 and asset_dim and returns_count:
            step_weights_array_tmp = np.zeros((returns_count, asset_dim), dtype=float)
        if asset_dim and step_weights_array_tmp.size:
            zero_row = np.zeros((1, step_weights_array_tmp.shape[1]), dtype=float)
            weights_arr_full = np.vstack([zero_row, step_weights_array_tmp])
        elif asset_dim and returns_count:
            weights_arr_full = np.vstack([np.zeros((1, asset_dim), dtype=float), np.zeros((returns_count, asset_dim), dtype=float)])
        else:
            weights_arr_full = np.empty((0, 0), dtype=float)

    asset_dim = weights_arr_full.shape[1] if weights_arr_full.size else asset_dim
    if weights_arr_full.size:
        if weights_arr_full.shape[0] >= returns_count + 1:
            step_weights_array = weights_arr_full[1 : 1 + returns_count]
        else:
            step_weights_array = weights_arr_full[-returns_count:]
    else:
        step_weights_array = np.asarray(step_weights, dtype=float)
        if step_weights_array.ndim == 1 and asset_dim:
            step_weights_array = step_weights_array.reshape(-1, asset_dim)
        if step_weights_array.size == 0 and asset_dim and returns_count:
            step_weights_array = np.zeros((returns_count, asset_dim), dtype=float)
        if asset_dim and step_weights_array.size and weights_arr_full.size == 0:
            zero_row = np.zeros((1, step_weights_array.shape[1]), dtype=float)
            weights_arr_full = np.vstack([zero_row, step_weights_array])

    if weights_arr_full.size == 0:
        unique_weights = 0
        total_dweight = 0.0
    else:
        rounded = np.round(weights_arr_full, 10)
        unique_weights = int(np.unique(rounded, axis=0).shape[0])
        diffs = np.diff(weights_arr_full, axis=0)
        total_dweight = float(np.abs(diffs).sum()) if diffs.size else 0.0

    env_index = None
    if hasattr(base_env_unwrapped, "index"):
        try:
            env_index = pd.DatetimeIndex(base_env_unwrapped.index)
        except Exception:
            env_index = None

    window_size = int(getattr(base_env_unwrapped, "window", 0))
    tf_minutes = _timeframe_minutes(cfg.get("timeframe", "5min"))
    if env_index is not None and returns_count:
        try:
            idx = pd.DatetimeIndex(env_index)
        except Exception:
            idx = None
        else:
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
        if idx is not None:
            start_pos = window_size + 1
            if start_pos + returns_count <= len(idx):
                return_timestamps = idx[start_pos : start_pos + returns_count]
            else:
                return_timestamps = idx[-returns_count:]
        else:
            return_timestamps = None
    else:
        return_timestamps = None

    if return_timestamps is None:
        if returns_count:
            freq = None
            if tf_minutes:
                freq = pd.Timedelta(minutes=int(tf_minutes))
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            if freq:
                return_timestamps = pd.date_range(start=start_ts + freq, periods=returns_count, freq=freq)
            else:
                return_timestamps = pd.date_range(start=start_ts, periods=returns_count, freq="D")
        else:
            return_timestamps = pd.DatetimeIndex([])

    if isinstance(return_timestamps, pd.DatetimeIndex):
        if return_timestamps.tz is None and len(return_timestamps):
            return_timestamps = return_timestamps.tz_localize("UTC")
        else:
            return_timestamps = return_timestamps.tz_convert("UTC")

    nav_series_for_csv: List[float]
    if len(equity_history) >= returns_count + 1:
        nav_series_for_csv = equity_history[1 : 1 + returns_count]
    else:
        nav_series_for_csv = equity_history[-returns_count:] if returns_count else []

    if step_weights_array.size and returns_count:
        prev_weights = weights_arr_full[:-1][-returns_count:] if weights_arr_full.shape[0] >= returns_count + 1 else np.vstack([np.zeros((1, step_weights_array.shape[1]), dtype=float), step_weights_array[:-1]])
        curr_weights = step_weights_array
        dweights_array = curr_weights - prev_weights
    else:
        dweights_array = np.empty_like(step_weights_array)

    calendar_key = calendar or cfg.get("calendar")
    metrics = compute(
        returns=returns,
        equity=equity_history,
        timeframe=cfg.get("timeframe", "5min"),
        symbols=cfg.get("symbols", []),
        calendar=calendar_key,
    )
    metrics["unique_weights"] = int(unique_weights)
    metrics["total_dweight"] = float(total_dweight)
    metrics["episode_resets"] = int(episode_resets)
    metrics["used_vecnormalize"] = bool(used_vecnormalize)
    if seed is not None:
        metrics["seed"] = int(seed)

    action_cfg = cfg.get("action", {})
    fields = EnvConfig.__dataclass_fields__
    thr = action_cfg.get("dweight_threshold")
    if thr is None:
        thr = cfg.get("dweight_threshold")
    if thr is None:
        thr = fields["dweight_threshold"].default
    thr = float(thr)
    trade_stats = _compute_trade_stats(weights_arr_full, thr, cfg.get("timeframe"), start, end)
    metrics.update(trade_stats)
    metrics["dweight_threshold"] = float(thr)

    max_dweight = action_cfg.get("max_dweight")
    if max_dweight is None:
        max_dweight = cfg.get("max_dweight")
    if max_dweight is None:
        max_dweight = fields["max_dweight"].default
    max_dweight = float(max_dweight)

    reward_cfg = cfg.get("reward", {})
    lambda_turnover = reward_cfg.get("lambda_turnover")
    if lambda_turnover is None:
        lambda_turnover = cfg.get("lambda_turnover")
    if lambda_turnover is None:
        lambda_turnover = fields["lambda_turnover"].default
    lambda_turnover = float(lambda_turnover)

    bars_count = returns_count
    bars_per_week = None
    if tf_minutes and tf_minutes > 0:
        bars_per_week = (7 * 24 * 60) / float(tf_minutes)

    out_dir.mkdir(parents=True, exist_ok=True)

    weight_payloads: List[Dict[str, Any]] = []
    if step_weights_array.size:
        for idx_step in range(returns_count):
            payload = {
                "target": step_weights_array[idx_step].tolist(),
                "delta": dweights_array[idx_step].tolist() if dweights_array.size else [],
            }
            weight_payloads.append(payload)
    else:
        for _ in range(returns_count):
            weight_payloads.append({"target": [], "delta": []})

    timestamps_output = []
    if isinstance(return_timestamps, pd.DatetimeIndex):
        if len(return_timestamps):
            ts_idx = return_timestamps
            if ts_idx.tz is None:
                ts_idx = ts_idx.tz_localize("UTC")
            else:
                ts_idx = ts_idx.tz_convert("UTC")
            timestamps_output = ts_idx.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

    if len(timestamps_output) != returns_count:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        freq = pd.to_timedelta(tf_minutes, unit='m') if tf_minutes else pd.Timedelta(days=1)
        timestamps_output = pd.date_range(start=start_ts, periods=returns_count, freq=freq).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ).tolist()

    returns_df = pd.DataFrame({"returns": returns, "ts": timestamps_output})
    returns_df.to_csv(out_dir / "oos_returns.csv", index=False)

    nav_df = pd.DataFrame({"equity": nav_series_for_csv, "ts": timestamps_output})
    nav_df.to_csv(out_dir / "oos_equity.csv", index=False)

    symbols = list(cfg.get("symbols", []))
    if (not symbols or len(symbols) != asset_dim) and hasattr(base_env_unwrapped, "symbols"):
        env_symbols = list(getattr(base_env_unwrapped, "symbols", []))
        if len(env_symbols) == asset_dim:
            symbols = env_symbols
    if len(symbols) != asset_dim:
        symbols = [f"asset_{idx}" for idx in range(asset_dim)]

    weight_records: List[Dict[str, Any]] = []
    for idx_step in range(returns_count):
        payload = weight_payloads[idx_step] if idx_step < len(weight_payloads) else {"target": [], "delta": []}
        weight_records.append({"weights": payload, "ts": timestamps_output[idx_step]})

    (out_dir / "oos_weights.json").write_text(json.dumps(weight_records, indent=2), encoding="utf-8")

    weight_rows_csv = [
        {"weights": json.dumps(record["weights"]), "ts": record["ts"]} for record in weight_records
    ]
    weights_df = pd.DataFrame(weight_rows_csv)
    weights_df.to_csv(out_dir / "oos_weights.csv", index=False)

    (out_dir / "oos_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _write_oos_meta(
        out_dir,
        cfg,
        model_path,
        vecnorm_path,
        start,
        end,
        deterministic,
        calendar_key,
        seed,
        bars=bars_count,
        bars_per_week=bars_per_week,
        dweight_threshold=thr,
        max_dweight=max_dweight,
        lambda_turnover=lambda_turnover,
        random_policy=random_policy,
    )

    required_outputs = [
        out_dir / "oos_returns.csv",
        out_dir / "oos_equity.csv",
        out_dir / "oos_weights.csv",
        out_dir / "oos_metrics.json",
        out_dir / "oos_meta.json",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required rollout outputs: {missing}")

    return metrics


__all__ = ["run_oos_eval"]
