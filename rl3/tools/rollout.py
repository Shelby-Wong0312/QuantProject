#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize

from eval.metrics import compute as compute_metrics
from eval.reports import write_markdown
from rl3.train.ppo_baseline import make_env

try:  # sb3-contrib is optional but required for recurrent PPO
    from sb3_contrib.ppo_recurrent import RecurrentPPO  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime with clear error
    RecurrentPPO = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL3 OOS rollout tool")
    parser.add_argument("--config", required=True, help="Training/eval config YAML")
    parser.add_argument("--model", help="Override path to saved model .zip")
    parser.add_argument("--vecnormalize", help="Override path to saved VecNormalize pickle")
    parser.add_argument("--out", help="Directory to write evaluation artifacts")
    parser.add_argument(
        "--device", default="auto", help="Torch device for inference (default: auto)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic policy during rollout (overrides config eval.deterministic)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Force stochastic policy during rollout (overrides config eval.deterministic)",
    )
    return parser.parse_args()


def _slug_ts(value: Any) -> str:
    ts = pd.Timestamp(value)
    return ts.strftime("%Y%m%d%H%M")


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_datestr(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")


def _parse_timedelta(value: Any) -> pd.Timedelta:
    if isinstance(value, (int, float)):
        return pd.Timedelta(days=float(value))
    if isinstance(value, str):
        try:
            return pd.Timedelta(value)
        except ValueError as exc:
            try:
                return pd.Timedelta(days=float(value))
            except ValueError as inner_exc:
                raise ValueError(f"Cannot parse timedelta from {value!r}") from inner_exc
    if isinstance(value, pd.Timedelta):
        return value
    raise TypeError(f"Unsupported timedelta spec: {value!r}")


def _build_windows(cfg: Dict[str, Any], eval_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    windows: List[Dict[str, str]] = []
    base_start = eval_cfg.get("start", cfg.get("start"))
    base_end = eval_cfg.get("end", cfg.get("end"))

    explicit = eval_cfg.get("windows")
    if explicit:
        for idx, raw in enumerate(explicit):
            start = raw.get("start")
            end = raw.get("end")
            if not start or not end:
                raise ValueError("Each eval window requires start and end")
            label = raw.get("label") or f"wf_{idx:02d}_{_slug_ts(start)}_{_slug_ts(end)}"
            windows.append({"label": label, "start": str(start), "end": str(end)})
        return windows

    rolling = eval_cfg.get("rolling")
    if rolling:
        start = _to_timestamp(rolling.get("start", base_start))
        end = _to_timestamp(rolling.get("end", base_end))
        if pd.isna(start) or pd.isna(end):
            raise ValueError("Rolling eval requires start and end")
        if start >= end:
            raise ValueError("Rolling eval requires start < end")
        window_spec = rolling.get("window") or rolling.get("window_days")
        if window_spec is None:
            raise ValueError("Rolling eval requires window/window_days")
        step_spec = rolling.get("step") or rolling.get("step_days") or window_spec
        window_td = _parse_timedelta(window_spec)
        step_td = _parse_timedelta(step_spec)
        prefix = rolling.get("label_prefix", "wf")

        idx = 0
        current_start = start
        while current_start < end:
            current_end = current_start + window_td
            if current_end > end:
                current_end = end
            label = f"{prefix}_{idx:02d}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            windows.append(
                {
                    "label": label,
                    "start": _to_datestr(current_start),
                    "end": _to_datestr(current_end),
                }
            )
            current_start = current_start + step_td
            idx += 1
        if not windows:
            raise ValueError("Rolling eval produced no windows")
        return windows

    if base_start and base_end:
        label = eval_cfg.get("label", "oos")
        windows.append({"label": label, "start": str(base_start), "end": str(base_end)})
        return windows

    raise ValueError("No evaluation window configured.")


def _unwrap_env(env: Any) -> Any:
    node = env
    visited: set[int] = set()
    while hasattr(node, "venv") and id(node) not in visited:
        visited.add(id(node))
        node = node.venv
    if hasattr(node, "envs") and node.envs:
        node = node.envs[0]
    while hasattr(node, "env"):
        node = node.env
    return node


def _load_algo_class(policy_name: str):
    if policy_name == "MlpLstmPolicy":
        if RecurrentPPO is None:
            raise RuntimeError(
                "sb3-contrib is required for recurrent PPO (pip install sb3-contrib)"
            )
        return RecurrentPPO
    return PPO


def _create_vec_env(cfg: Dict[str, Any], vec_path: Optional[Path]) -> Tuple[Any, bool]:
    base_cfg = copy.deepcopy(cfg)
    factory = lambda cfg_copy=base_cfg: make_env(cfg_copy)
    env = DummyVecEnv([factory])
    used_norm = False
    if vec_path is not None:
        try:
            vec_env = VecNormalize.load(str(vec_path), env)
        except Exception as exc:
            raise RuntimeError(f"Failed to load VecNormalize stats from {vec_path}: {exc}") from exc
        vec_env.training = False
        vec_env.norm_reward = False
        used_norm = True
    else:
        vec_env = env


def _run_window(
    window: Dict[str, str],
    cfg: Dict[str, Any],
    algo_cls,
    model_path: Path,
    vec_path: Optional[Path],
    deterministic: bool,
    device: str,
) -> Dict[str, Any]:
    window_cfg = copy.deepcopy(cfg)
    window_cfg["start"] = window["start"]
    window_cfg["end"] = window["end"]

    vec_env, used_norm = _create_vec_env(window_cfg, vec_path)

    model = algo_cls.load(str(model_path), env=vec_env, device=device)
    model.policy.set_training_mode(False)

    obs = vec_env.reset()
    state = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    rewards: List[float] = []

    while True:
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, reward, dones, infos = vec_env.step(action)
        rewards.append(float(reward[0]))
        episode_starts = dones
        if bool(dones[0]):
            break

    base_env = _unwrap_env(vec_env)
    if hasattr(base_env, "get_episode_trajectory"):
        traj = base_env.get_episode_trajectory()
    else:
        equity = [1.0]
        for r in rewards:
            equity.append(equity[-1] * (1.0 + r))
        traj = {"returns": rewards, "equity": equity, "weights": []}

    vec_env.close()

    returns = list(map(float, traj.get("returns", [])))
    equity = list(map(float, traj.get("equity", [])))
    weights = traj.get("weights", [])

    metrics = compute_metrics(
        returns=returns,
        equity=equity,
        timeframe=cfg.get("timeframe", "5min"),
        symbols=cfg.get("symbols", []),
    )

    final_equity = equity[-1] if equity else 1.0
    return {
        "label": window["label"],
        "start": window["start"],
        "end": window["end"],
        "steps": len(returns),
        "returns": returns,
        "equity": equity,
        "weights": weights,
        "metrics": metrics,
        "final_nav": final_equity,
        "used_vecnormalize": used_norm,
        "deterministic": deterministic,
    }


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("Config root must be a mapping")

    train_cfg = cfg.get("train", {})
    policy_name = train_cfg.get("policy", "MlpLstmPolicy")
    algo_cls = _load_algo_class(policy_name)

    eval_cfg = cfg.get("eval", {})
    base_log_dir = Path(cfg.get("log", {}).get("dir", "runs/rl3/baseline"))

    model_path = Path(args.model or eval_cfg.get("model_path", base_log_dir / "model.zip"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    vec_path_raw = (
        args.vecnormalize
        or eval_cfg.get("vecnormalize_path")
        or (base_log_dir / "vecnormalize.pkl")
    )
    vec_path = Path(vec_path_raw) if vec_path_raw else None
    vec_path_display = str(vec_path_raw) if vec_path_raw else None

    out_dir = Path(args.out or eval_cfg.get("dir", base_log_dir / "oos"))
    out_dir.mkdir(parents=True, exist_ok=True)

    deterministic = eval_cfg.get("deterministic", True)
    if args.deterministic:
        deterministic = True
    if args.stochastic:
        deterministic = False

    windows = _build_windows(cfg, eval_cfg)

    print(
        f"Loaded config from {cfg_path}. Running {len(windows)} window(s) using model {model_path}."
    )

    vecnormalize_found = False
    if vec_path is not None:
        if vec_path.exists():
            vecnormalize_found = True
        else:
            print(
                f"Warning: VecNormalize file not found at {vec_path}, proceeding without normalization."
            )
            vec_path = None

    results: List[Dict[str, Any]] = []
    agg_returns: List[float] = []
    agg_equity: List[float] = [1.0]

    for window in windows:
        print(f"-> Window {window['label']} [{window['start']} -> {window['end']}] ...", end=" ")
        res = _run_window(window, cfg, algo_cls, model_path, vec_path, deterministic, args.device)
        results.append(res)
        for r in res["returns"]:
            agg_returns.append(r)
            agg_equity.append(agg_equity[-1] * (1.0 + r))
        metrics = res["metrics"]
        print(
            f"done. Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_drawdown']:.2%}, NAV={res['final_nav']:.3f}"
        )

        win_dir = out_dir / res["label"]
        win_dir.mkdir(parents=True, exist_ok=True)
        (win_dir / "returns.json").write_text(json.dumps(res["returns"]), encoding="utf-8")
        (win_dir / "equity.json").write_text(json.dumps(res["equity"]), encoding="utf-8")
        (win_dir / "weights.json").write_text(json.dumps(res.get("weights", [])), encoding="utf-8")
        (win_dir / "metrics.json").write_text(
            json.dumps(res["metrics"], indent=2), encoding="utf-8"
        )
        meta_payload = {
            k: res[k]
            for k in (
                "label",
                "start",
                "end",
                "steps",
                "deterministic",
                "used_vecnormalize",
                "final_nav",
            )
        }
        (win_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        write_markdown(win_dir / "report.md", res["metrics"], cfg.get("symbols", []))

    overall_metrics = compute_metrics(
        returns=agg_returns,
        equity=agg_equity,
        timeframe=cfg.get("timeframe", "5min"),
        symbols=cfg.get("symbols", []),
    )

    summary = {
        "config": str(cfg_path),
        "model": str(model_path),
        "vecnormalize": vec_path_display,
        "vecnormalize_found": vecnormalize_found,
        "deterministic": deterministic,
        "windows": [
            {
                "label": r["label"],
                "start": r["start"],
                "end": r["end"],
                "steps": r["steps"],
                "metrics": r["metrics"],
                "final_nav": r["final_nav"],
                "used_vecnormalize": r["used_vecnormalize"],
                "deterministic": r["deterministic"],
            }
            for r in results
        ],
        "aggregate_metrics": overall_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", overall_metrics, cfg.get("symbols", []))
    print(
        f"Finished. Aggregate Sharpe={overall_metrics['sharpe']:.3f}, MaxDD={overall_metrics['max_drawdown']:.2%}."
    )


if __name__ == "__main__":
    main()
