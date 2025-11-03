#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from rl3.eval.rollout import run_oos_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic OOS rollout aligned with score_oos_gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Training/eval config file (YAML or JSON).")
    parser.add_argument("--model", help="Override path to trained model .zip file.")
    parser.add_argument("--vecnormalize", help="Override path to VecNormalize pickle file.")
    parser.add_argument("--out", help="Directory to write rollout artifacts (root for wf_*).")
    parser.add_argument("--start", help="Single-window start override (ISO format).")
    parser.add_argument("--end", help="Single-window end override (ISO format).")
    parser.add_argument("--label", help="Label for single-window runs (default auto wf label).")
    parser.add_argument(
        "--prefix",
        default="wf",
        help="Prefix for auto-generated window labels (ignored if config provides labels).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Run rollout with stochastic policy instead of deterministic.",
    )
    return parser.parse_args()


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping.")
    return data


def _resolve_path(spec: str | Path | None, base_dir: Path) -> Path | None:
    if spec is None:
        return None
    path = Path(spec).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_cli_path(spec: str | Path | None) -> Path | None:
    if spec is None:
        return None
    path = Path(spec).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _slug_ts(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%d")


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_timedelta(value: Any) -> pd.Timedelta:
    if isinstance(value, (int, float)):
        return pd.Timedelta(days=float(value))
    if isinstance(value, str):
        try:
            return pd.Timedelta(value)
        except Exception:
            return pd.to_timedelta(float(value), unit="D")
    if isinstance(value, pd.Timedelta):
        return value
    raise TypeError(f"Unsupported timedelta spec: {value!r}")


def _build_windows(
    cfg: Dict[str, Any],
    start_override: str | None,
    end_override: str | None,
    label_override: str | None,
    prefix: str,
) -> List[Dict[str, str]]:
    eval_cfg = cfg.get("eval", {})
    windows: List[Dict[str, str]] = []

    if start_override or end_override:
        if not start_override or not end_override:
            raise ValueError("Both --start and --end must be provided when overriding window.")
        label = label_override or f"{prefix}_00_{_slug_ts(start_override)}_{_slug_ts(end_override)}"
        windows.append({"label": label, "start": str(start_override), "end": str(end_override)})
        return windows

    explicit = eval_cfg.get("windows")
    if explicit:
        for idx, raw in enumerate(explicit):
            start = raw.get("start")
            end = raw.get("end")
            if not start or not end:
                raise ValueError("Each eval window requires start and end.")
            label = raw.get("label")
            if not label:
                label = f"{prefix}_{idx:02d}_{_slug_ts(start)}_{_slug_ts(end)}"
            elif not label.startswith("wf_"):
                label = label  # respect user label even if it does not start with wf_
            windows.append({"label": str(label), "start": str(start), "end": str(end)})
        return windows

    rolling = eval_cfg.get("rolling")
    if rolling:
        base_start = rolling.get("start") or cfg.get("start")
        base_end = rolling.get("end") or cfg.get("end")
        start_ts = _to_timestamp(base_start)
        end_ts = _to_timestamp(base_end)
        if start_ts >= end_ts:
            raise ValueError("Rolling eval requires start < end.")
        window_spec = rolling.get("window") or rolling.get("window_days")
        step_spec = rolling.get("step") or rolling.get("step_days") or window_spec
        window_td = _parse_timedelta(window_spec)
        step_td = _parse_timedelta(step_spec)
        label_prefix = rolling.get("label_prefix", prefix)
        idx = 0
        current_start = start_ts
        while current_start < end_ts:
            current_end = current_start + window_td
            if current_end > end_ts:
                current_end = end_ts
            label = f"{label_prefix}_{idx:02d}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            windows.append(
                {
                    "label": label,
                    "start": current_start.isoformat(),
                    "end": current_end.isoformat(),
                }
            )
            current_start = current_start + step_td
            idx += 1
        if not windows:
            raise ValueError("Rolling eval produced no windows.")
        return windows

    base_start = eval_cfg.get("start") or cfg.get("start")
    base_end = eval_cfg.get("end") or cfg.get("end")
    if base_start and base_end:
        label = eval_cfg.get("label") or f"{prefix}_00_{_slug_ts(base_start)}_{_slug_ts(base_end)}"
        windows.append({"label": str(label), "start": str(base_start), "end": str(base_end)})
        return windows

    raise ValueError("No evaluation window defined. Provide --start/--end or config eval windows.")


def _ensure_wf_prefix(label: str, prefix: str) -> str:
    if label.startswith("wf_"):
        return label
    return f"{prefix}_{label}"


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = _load_config(cfg_path)
    cfg_dir = cfg_path.parent
    eval_cfg = cfg.get("eval", {})
    log_cfg = cfg.get("log", {})

    base_log_dir = _resolve_path(log_cfg.get("dir") or "runs/rl3/baseline", cfg_dir)

    model_path = _resolve_cli_path(args.model)
    if model_path is None:
        model_path = _resolve_path(eval_cfg.get("model_path"), cfg_dir)
    if model_path is None and base_log_dir is not None:
        model_path = (base_log_dir / "model.zip").resolve()
    if model_path is None:
        raise ValueError("Unable to determine model path; supply --model or set eval.model_path.")

    vec_path = _resolve_cli_path(args.vecnormalize)
    if vec_path is None:
        vec_path = _resolve_path(eval_cfg.get("vecnormalize_path"), cfg_dir)
    if vec_path is None and base_log_dir is not None:
        vec_path = (base_log_dir / "vecnormalize.pkl").resolve()
    if vec_path is None:
        raise ValueError(
            "Unable to determine VecNormalize path; supply --vecnormalize or set eval.vecnormalize_path."
        )

    out_root = _resolve_cli_path(args.out)
    if out_root is None:
        out_root = _resolve_path(eval_cfg.get("dir"), cfg_dir)
    if out_root is None and base_log_dir is not None:
        out_root = (base_log_dir / "oos").resolve()
    if out_root is None:
        out_root = (Path.cwd() / "tmp_oos").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    windows = _build_windows(cfg, args.start, args.end, args.label, args.prefix)

    deterministic = not args.stochastic
    summary_windows: List[Dict[str, Any]] = []

    for idx, window in enumerate(windows):
        label = _ensure_wf_prefix(window["label"], args.prefix)
        window_dir = out_root / label
        oos_dir = window_dir / "oos"
        window_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{idx+1}/{len(windows)}] Window {label}: {window['start']} -> {window['end']} "
            f"(deterministic={deterministic})"
        )
        metrics = run_oos_eval(
            model_path=model_path,
            vecnorm_path=vec_path,
            cfg_path=cfg_path,
            start=window["start"],
            end=window["end"],
            out_dir=oos_dir,
            deterministic=deterministic,
        )
        sharpe = float(metrics.get("sharpe", 0.0))
        max_dd = float(metrics.get("max_drawdown", 0.0))
        print(f"    -> Sharpe={sharpe:.3f}, MaxDD={max_dd:.2%}, artifacts: {oos_dir}")

        summary_windows.append(
            {
                "label": label,
                "start": window["start"],
                "end": window["end"],
                "metrics": metrics,
                "oos_dir": str(oos_dir),
            }
        )

    summary_payload = {
        "config": str(cfg_path),
        "model": str(model_path),
        "vecnormalize": str(vec_path),
        "out_root": str(out_root),
        "deterministic": deterministic,
        "windows": summary_windows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote summary -> {out_root / 'summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
