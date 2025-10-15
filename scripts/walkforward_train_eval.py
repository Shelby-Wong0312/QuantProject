from __future__ import annotations

import argparse
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import json
import shutil
import time
from pathlib import Path
from typing import List, Tuple
import yaml
import numpy as np
from pandas import Timestamp as _PDTimestamp, Timedelta as _PDTimedelta
from pandas.tseries.offsets import BDay as _PDBDay


def _timeframe_minutes(timeframe: str | None) -> float | None:
    if timeframe is None:
        return None
    tf = str(timeframe).strip().lower()
    if not tf:
        return None
    if tf in {"1d", "day", "daily"}:
        return 1440.0
    for suffix, multiplier in (("min", 1.0), ("m", 1.0), ("h", 60.0)):
        if tf.endswith(suffix):
            token = tf[: -len(suffix)] or "1"
            try:
                return float(token) * multiplier
            except ValueError:
                return None
    try:
        return float(tf)
    except ValueError:
        return None


def _to_utc(ts_str: str) -> _PDTimestamp:
    ts = _PDTimestamp(ts_str)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl3.eval.rollout import run_oos_eval, _make_env
from rl3.env.portfolio_env import EnvConfig
from rl3.train.ppo_baseline import main as train_main
from src.quantproject.data_pipeline.loaders.bars import load_and_align_router


MIN_TRAIN_BUSINESS_DAYS = 12


FEATURE_WARMUP = {
    "logret": 1,
    "rsi": 14,
    "macd": 26,
    "atr": 14,
    "vol_ewma": 20,
    "vol_zscore": 20,
    "range_pct": 1,
    "trend_persist": 10,
    "breakout_flags": 20,
}


def _max_feature_warmup(cfg: dict) -> int:
    obs_cfg = cfg.get("obs") or {}
    features = list(obs_cfg.get("features") or cfg.get("features") or [])
    if not features:
        return 0
    override = obs_cfg.get("warmup_override") if isinstance(obs_cfg, dict) else None
    if override is None:
        override = cfg.get("feature_warmup_override")
    if override is not None:
        try:
            return max(0, int(override))
        except (TypeError, ValueError):
            pass
    return max(FEATURE_WARMUP.get(name, 0) for name in features)


def _ensure_min_train_span(
    train_start: str, train_end: str, test_end: str, min_bdays: int = MIN_TRAIN_BUSINESS_DAYS
):
    if min_bdays <= 0:
        return train_start, train_end, False
    start_ts = _to_utc(train_start)
    end_ts = _to_utc(train_end)
    test_end_ts = _to_utc(test_end)
    span = int(np.busday_count(start_ts.date(), (end_ts + _PDTimedelta(days=1)).date()))
    if span >= min_bdays:
        return train_start, train_end, False
    needed = min_bdays - span
    extended_end = end_ts + _PDBDay(needed)
    if extended_end >= test_end_ts:
        raise ValueError(
            f"Cannot extend training window ({train_start}~{train_end}) by {needed} business days without reaching test_end {test_end}."
        )
    return train_start, extended_end.date().isoformat(), True


def _date_range_pairs(
    train_start: str, train_end: str, test_end: str, step_days: int = 5
) -> List[Tuple[str, str, str]]:

    t0 = _PDTimestamp(train_start).tz_localize("UTC")
    t1 = _PDTimestamp(train_end).tz_localize("UTC")
    te = _PDTimestamp(test_end).tz_localize("UTC")
    out: List[Tuple[str, str, str]] = []
    while t1 < te:
        test_start = t1
        test_stop = min(t1 + _PDTimedelta(days=step_days), te)
        out.append((t0.date().isoformat(), t1.date().isoformat(), test_stop.date().isoformat()))
        t0 += _PDTimedelta(days=step_days)
        t1 += _PDTimedelta(days=step_days)
    return out


def _deepcopy_cfg(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg))


def _ensure_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    legacy = model_path.parent / "ppo_model.zip"
    if legacy.exists():
        shutil.copyfile(legacy, model_path)
        return model_path
    raise FileNotFoundError(f"Model artifact missing: {model_path}")


def run_wf(
    cfg_path: Path,
    out_root: Path,
    train_start: str,
    train_end: str,
    test_end: str,
    step_days: int = 7,
    save_oos: bool = False,
) -> List[dict]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    train_start, train_end, _span_adjusted = _ensure_min_train_span(
        train_start, train_end, test_end
    )
    if _span_adjusted:
        print(
            f"[WF] Extended training window end to {train_end} to satisfy {MIN_TRAIN_BUSINESS_DAYS} business-day minimum."
        )
    if not isinstance(cfg, dict):
        raise TypeError("Training config must be a mapping.")

    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out_root / cfg_path.name)

    windows = _date_range_pairs(train_start, train_end, test_end, step_days)
    if not windows:
        raise ValueError("No walkforward windows generated. Check date inputs.")

    base_timeframe = cfg.get("timeframe")
    timeframe_minutes = _timeframe_minutes(base_timeframe)

    results: List[dict] = []
    detailed: List[dict] = []
    skipped_windows: List[dict] = []
    for idx, (tr_s, tr_e, te_e) in enumerate(windows, start=1):
        oos_start = (_to_utc(tr_e) + _PDTimedelta(days=1)).date().isoformat()
        oos_start_ts = _to_utc(oos_start)
        oos_end_ts = _to_utc(te_e)
        span_seconds = max(0.0, (oos_end_ts - oos_start_ts).total_seconds())
        skip_reason: str | None = None
        if span_seconds < 24 * 3600:
            skip_reason = f"OOS span {span_seconds / 3600:.2f}h < 24h"
        elif timeframe_minutes:
            est_bars = span_seconds / (timeframe_minutes * 60.0)
            if est_bars < 3.0:
                skip_reason = f"Estimated OOS bars {est_bars:.2f} < 3"

        if skip_reason:
            print(
                f"Skipping window {idx:02d} (train {tr_s}~{tr_e}, OOS {oos_start}~{te_e}): {skip_reason}"
            )
            skipped_windows.append(
                {
                    "index": idx,
                    "train_start": tr_s,
                    "train_end": tr_e,
                    "test_start": oos_start,
                    "test_end": te_e,
                    "reason": skip_reason,
                }
            )
            continue

        run_dir = out_root / f"wf_{idx:02d}"
        train_dir = run_dir / "train"
        oos_dir = run_dir / "oos"
        train_dir.mkdir(parents=True, exist_ok=True)
        oos_dir.mkdir(parents=True, exist_ok=True)

        wf_cfg = _deepcopy_cfg(cfg)
        wf_cfg["start"], wf_cfg["end"] = tr_s, tr_e
        symbols_cfg = list(wf_cfg.get("symbols", []))
        timeframe_cfg = wf_cfg.get("timeframe", cfg.get("timeframe", "5min"))
        obs_cfg = wf_cfg.get("obs") or {}
        window_default = EnvConfig.__dataclass_fields__["window"].default
        window = int(obs_cfg.get("window", wf_cfg.get("window", window_default)))
        feature_warmup = _max_feature_warmup(wf_cfg)
        required_bars = window + 1 + feature_warmup
        try:
            data_probe = load_and_align_router(symbols_cfg, tr_s, tr_e, timeframe_cfg)
        except Exception as exc:
            reason = f"Failed to fetch aligned bars: {exc}"
            (run_dir / "SKIPPED_TRAIN_LOAD").write_text(reason, encoding="utf-8")
            skipped_windows.append(
                {
                    "index": idx,
                    "train_start": tr_s,
                    "train_end": tr_e,
                    "test_start": oos_start,
                    "test_end": te_e,
                    "reason": reason,
                }
            )
            continue
        train_len = min((len(df) for df in data_probe.values())) if data_probe else 0
        if train_len <= 0:
            print(
                f"[WF] {idx:02d} train {tr_s}~{tr_e} | train_bars_raw=0, train_bars_eff=0, window={window}, warmup={feature_warmup} -> SKIP (no data)"
            )
        else:
            train_bars_raw = train_len
            train_bars_effective = max(0, train_len - feature_warmup)
            verdict = "OK" if train_len >= required_bars else "SKIP"
            print(
                f"[WF] {idx:02d} train {tr_s}~{tr_e} | train_bars_raw={train_bars_raw}, "
                f"train_bars_eff={train_bars_effective}, window={window}, warmup={feature_warmup} -> {verdict}"
            )
        if train_len <= 0:
            print(f"[WF] {idx:02d} train {tr_s}~{tr_e} -> no aligned bars returned")
        else:
            bar_note = f"window={window}, warmup={feature_warmup}, required>={required_bars}"
            print(f"[WF] {idx:02d} train {tr_s}~{tr_e} -> bars={train_len} ({bar_note})")
        if train_len < required_bars:
            reason = f"Train window bars {train_len} < required {required_bars} (window={window}, warmup={feature_warmup})"
            (run_dir / "SKIPPED_TRAIN_TOO_SHORT").write_text(reason, encoding="utf-8")
            skipped_windows.append(
                {
                    "index": idx,
                    "train_start": tr_s,
                    "train_end": tr_e,
                    "test_start": oos_start,
                    "test_end": te_e,
                    "reason": reason,
                }
            )
            continue

        wf_cfg.setdefault("log", {})["dir"] = str(train_dir)

        tmp_yaml = run_dir / "train.yaml"
        tmp_yaml.write_text(
            yaml.safe_dump(wf_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        t0 = time.time()
        try:
            train_main(str(tmp_yaml))
        except ValueError as exc:
            msg = str(exc).strip().replace("\n", " ")
            value_error_signals = ("Not enough bars for window", "requires at least")
            if any(token in msg for token in value_error_signals):
                reason = msg
                (run_dir / "SKIPPED_TRAIN_TOO_SHORT").write_text(reason, encoding="utf-8")
                skipped_windows.append(
                    {
                        "index": idx,
                        "train_start": tr_s,
                        "train_end": tr_e,
                        "test_start": oos_start,
                        "test_end": te_e,
                        "reason": "SKIPPED_TRAIN_TOO_SHORT",
                        "detail": reason,
                    }
                )
                continue
            raise
        except AssertionError as exc:
            msg = str(exc).strip().replace("\n", " ")
            assertion_signals = (
                "Not enough bars",
                "No symbols returned data",
                "Unable to align data",
            )
            if any(token in msg for token in assertion_signals):
                reason = msg
                (run_dir / "SKIPPED_TRAIN_TOO_SHORT").write_text(reason, encoding="utf-8")
                skipped_windows.append(
                    {
                        "index": idx,
                        "train_start": tr_s,
                        "train_end": tr_e,
                        "test_start": oos_start,
                        "test_end": te_e,
                        "reason": "SKIPPED_TRAIN_TOO_SHORT",
                        "detail": reason,
                    }
                )
                continue
            raise
        t1 = time.time()
        train_elapsed = t1 - t0

        model_path = _ensure_model_path(train_dir / "model.zip")
        vecnorm_path = train_dir / "vecnormalize.pkl"
        cfg_json = train_dir / "config.json"
        if not vecnorm_path.exists():
            raise FileNotFoundError(f"VecNormalize artifact missing: {vecnorm_path}")

        cfg_for_eval = cfg_json if cfg_json.exists() else cfg_path

        base_env = None
        try:
            base_env = _make_env(wf_cfg, tr_e, te_e)
            oos_len = len(base_env.index) if hasattr(base_env, "index") else 0
        except ValueError as exc:
            reason = f"OOS env error: {exc}"
            (run_dir / "SKIPPED_OOS_INIT").write_text(reason, encoding="utf-8")
            skipped_windows.append(
                {
                    "index": idx,
                    "train_start": tr_s,
                    "train_end": tr_e,
                    "test_start": oos_start,
                    "test_end": te_e,
                    "reason": reason,
                }
            )
            continue
        finally:
            try:
                if base_env is not None:
                    base_env.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if (_PDTimestamp(te_e) - _PDTimestamp(tr_e)).total_seconds() < 24 * 3600 or oos_len < 3:
            (run_dir / "SKIPPED_TOO_SHORT").write_text(
                "OOS window < 1 day or < 3 bars", encoding="utf-8"
            )
            skipped_windows.append(
                {
                    "index": idx,
                    "train_start": tr_s,
                    "train_end": tr_e,
                    "test_start": oos_start,
                    "test_end": te_e,
                    "reason": "OOS <24h or <3 bars",
                }
            )
            continue

        metrics = run_oos_eval(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            cfg_path=cfg_for_eval,
            start=oos_start,
            end=te_e,
            out_dir=oos_dir,
        )

        window_result = {
            "i": idx,
            "train": [tr_s, tr_e],
            "test": [oos_start, te_e],
            "seconds": train_elapsed,
            **metrics,
        }
        results.append(window_result)

        detailed_result = {
            "index": idx,
            "train_start": tr_s,
            "train_end": tr_e,
            "test_start": oos_start,
            "test_end": te_e,
            "train_seconds": train_elapsed,
            "metrics": metrics,
            "train_dir": str(train_dir),
            "oos_dir": str(oos_dir),
            "oos_unique_weights": int(metrics.get("unique_weights", 0)),
            "oos_total_dweight": float(metrics.get("total_dweight", 0.0)),
            "oos_episode_resets": int(metrics.get("episode_resets", 1)),
        }
        (run_dir / "result.json").write_text(
            json.dumps(detailed_result, indent=2), encoding="utf-8"
        )
        detailed.append(detailed_result)

    summary_payload = {
        "include_oos_payload": bool(save_oos),
        "config": str(cfg_path),
        "train_start": train_start,
        "train_end": train_end,
        "test_end": test_end,
        "step_days": step_days,
        "windows": detailed,
    }
    if skipped_windows:
        summary_payload["skipped_windows"] = skipped_windows
    (out_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    lines = ["# RL3  WalkForward Summary", ""]
    for r in results:
        lines.extend(
            [
                f"## Window {r['i']:02d}  ({r['train'][0]}  {r['train'][1]}  |  OOS {r['test'][0]}  {r['test'][1]})",
                f"- Annualized Return: {r['ann_return']:.4f}",
                f"- Annualized Vol: {r['ann_vol']:.4f}",
                f"- Sharpe: {r['sharpe']:.3f}",
                f"- Max Drawdown: {r['max_drawdown']:.3%}",
                "",
            ]
        )
    if skipped_windows:
        lines.append("## Skipped Windows")
        for s in skipped_windows:
            detail = s.get("detail")
            reason_text = s["reason"] if detail is None else f"{s['reason']} - {detail}"
            lines.append(
                f"- {s['index']:02d}: train {s['train_start']}~{s['train_end']} | OOS {s['test_start']}~{s['test_end']} ({reason_text})"
            )
        lines.append("")
    (out_root / "WF_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    return detailed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward training and OOS evaluation")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--out", required=True, help="Directory to write walk-forward artifacts")
    parser.add_argument(
        "--train-start", required=True, help="Initial training window start (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--train-end", required=True, help="Initial training window end (YYYY-MM-DD)"
    )
    parser.add_argument("--test-end", required=True, help="Final OOS window end (YYYY-MM-DD)")
    parser.add_argument(
        "--step-days", type=int, default=5, help="Step size in days between windows"
    )
    parser.add_argument(
        "--save-oos", action="store_true", help="Keep OOS equity/returns even if already present"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_wf(
        cfg_path=Path(args.config),
        out_root=Path(args.out),
        train_start=args.train_start,
        train_end=args.train_end,
        test_end=args.test_end,
        step_days=args.step_days,
        save_oos=args.save_oos,
    )


if __name__ == "__main__":
    main()
