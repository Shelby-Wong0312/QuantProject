#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from rl3.eval.io import build_data_router, build_eval_windows, load_config
from rl3.eval.rollout import run_oos_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic/stochastic OOS rollout aligned with score_oos_gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="Path to Stable-Baselines3 model.zip.")
    parser.add_argument(
        "--vecnorm", help="Optional path to VecNormalize pickle. If missing, runs without normalization."
    )
    parser.add_argument(
        "--env-config",
        required=True,
        help="Environment configuration (YAML/JSON) describing symbols/features/eval windows.",
    )
    parser.add_argument(
        "--data-config",
        help="Optional data router configuration (YAML/JSON) for DataRouter backends.",
    )
    parser.add_argument(
        "--window",
        help="Override evaluation window in 'YYYY-MM-DD,YYYY-MM-DD' format (inclusive start/end).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory where wf_* subfolders and oos artifacts will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic rollout (use --no-deterministic to allow stochastic).",
    )
    parser.add_argument(
        "--calendar",
        choices=["equity", "crypto"],
        default="equity",
        help="Annualization calendar to use for metrics.",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Use random actions instead of a trained model (ignores --model/--vecnorm).",
    )
    return parser.parse_args()


def _seed_environment(seed: int | None) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    args = _parse_args()
    _seed_environment(args.seed)

    env_cfg = load_config(args.env_config)
    data_cfg = load_config(args.data_config) if args.data_config else None
    router = build_data_router(data_cfg)

    out_root = Path(args.outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    windows = build_eval_windows(env_cfg, args.window, prefix="wf")
    summaries: List[Dict[str, Any]] = []

    if args.random_policy:
        if args.model:
            raise SystemExit("--random-policy cannot be combined with --model.")
        if args.vecnorm:
            raise SystemExit("--random-policy cannot be combined with --vecnorm.")
    elif not args.model:
        raise SystemExit("--model is required unless --random-policy is specified.")

    model_path = Path(args.model).expanduser().resolve() if args.model else None
    vecnorm_path = Path(args.vecnorm).expanduser().resolve() if args.vecnorm else None
    env_config_path = Path(args.env_config).expanduser().resolve()
    data_config_path = Path(args.data_config).expanduser().resolve() if args.data_config else None

    for idx, window in enumerate(windows, start=1):
        label = window["label"]
        wf_dir = out_root / label
        oos_dir = wf_dir / "oos"
        if wf_dir.exists():
            shutil.rmtree(wf_dir)
        oos_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{idx}/{len(windows)}] Window {label}: {window['start']} -> {window['end']} "
            f"(deterministic={args.deterministic}, random_policy={args.random_policy})"
        )
        metrics = run_oos_eval(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            cfg_path=env_config_path,
            start=window["start"],
            end=window["end"],
            out_dir=oos_dir,
            deterministic=args.deterministic,
            cfg_override=env_cfg,
            data_router=router,
            calendar=args.calendar,
            seed=args.seed,
            random_policy=args.random_policy,
        )
        sharpe = float(metrics.get("sharpe", 0.0))
        max_dd = float(metrics.get("max_drawdown", 0.0))
        print(f"    -> Sharpe={sharpe:.3f}, MaxDD={max_dd:.2%}, artifacts: {oos_dir}")
        summaries.append(
            {
                "label": label,
                "start": window["start"],
                "end": window["end"],
                "metrics": metrics,
                "oos_dir": str(oos_dir),
                "random_policy": args.random_policy,
            }
        )

    summary_payload = {
        "model": str(model_path) if model_path else None,
        "vecnormalize": str(vecnorm_path) if vecnorm_path else None,
        "env_config": str(env_config_path),
        "data_config": str(data_config_path) if data_config_path else None,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "calendar": args.calendar,
        "random_policy": args.random_policy,
        "windows": summaries,
    }
    (out_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote summary -> {out_root / 'summary.json'}")
    print(f"Artifacts written to: {out_root}")


if __name__ == "__main__":
    main()
