from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("empty iso timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.fromisoformat(f"{value}T00:00:00+00:00")


def _weeks_from_meta(oos_dir: Path) -> float | None:
    meta_path = oos_dir / "oos_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta_data = _load_json(meta_path)
    except json.JSONDecodeError:
        return None
    start_raw = meta_data.get("start")
    end_raw = meta_data.get("end")
    if not start_raw or not end_raw:
        return None
    try:
        start = _parse_iso(str(start_raw))
        end = _parse_iso(str(end_raw))
    except Exception:
        return None
    seconds = max(0.0, (end - start).total_seconds())
    return max(1e-9, seconds / (7 * 24 * 3600.0))


def _prepare_weights(oos_dir: Path) -> np.ndarray:
    weights_raw = _load_json(oos_dir / "oos_weights.json")
    weights = np.asarray(weights_raw, dtype=float)
    if weights.ndim == 0:
        weights = weights.reshape(1, 1)
    elif weights.ndim == 1:
        weights = weights[:, np.newaxis]
    return weights


def _count_trades(weights: np.ndarray, threshold: float) -> tuple[int, int, np.ndarray]:
    rows = weights.shape[0]
    cols = weights.shape[1] if weights.ndim > 1 else 1
    if rows < 2:
        empty = np.zeros((0, max(cols, 1)), dtype=float)
        return 0, 0, empty
    diffs = np.abs(np.diff(weights, axis=0))
    if diffs.ndim == 1:
        diffs = diffs[:, np.newaxis]
    diffs = np.nan_to_num(diffs, nan=0.0)
    step_hits = int((diffs >= threshold).any(axis=1).sum())
    accum_vec = np.zeros(diffs.shape[1], dtype=float)
    accum_hits = 0
    for row in diffs:
        accum_vec += row
        mask = accum_vec >= threshold
        if mask.any():
            accum_hits += 1
            accum_vec[mask] = 0.0
    return accum_hits, step_hits, diffs


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))


def _evaluate_gate(
    entry: dict[str, Any],
    gate_by: str,
    sharpe_min: float,
    maxdd_max: float,
    tpw_max: float,
) -> tuple[str, str]:
    sharpe = entry["sharpe"]
    max_dd = abs(entry["max_drawdown"])
    tpw = entry["tpw_accum"] if gate_by == "accum" else entry["tpw_step"]

    reasons: list[str] = []
    if np.isnan(sharpe) or sharpe < sharpe_min:
        reasons.append(f"sharpe<{sharpe_min:.2f}")
    if np.isnan(max_dd) or max_dd > maxdd_max:
        reasons.append(f"maxdd>{maxdd_max:.2%}")
    if np.isnan(tpw) or tpw > tpw_max:
        reasons.append(f"trades/week>{tpw_max:.1f}")

    status = "PASS" if not reasons else "FAIL"
    detail = "" if not reasons else ";".join(reasons)
    return status, detail


def _collect(
    oos_dir: Path,
    threshold: float,
    bars_per_week: float | None,
    min_weeks: float,
) -> dict[str, Any] | None:
    metrics_path = oos_dir / "oos_metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _load_json(metrics_path)
    weights = _prepare_weights(oos_dir)
    accum_hits, step_hits, diffs = _count_trades(weights, threshold)
    total_dweight = float(np.nansum(diffs)) if diffs.size else 0.0

    bpw = bars_per_week if bars_per_week and bars_per_week > 0 else 168.0
    weeks_from_bars = max(1e-9, (weights.shape[0] / bpw) if weights.shape[0] else 0.0)
    weeks_from_meta = _weeks_from_meta(oos_dir) or 0.0
    weeks = max(weeks_from_meta, weeks_from_bars)
    if weeks < min_weeks:
        return None

    tpw_accum = accum_hits / weeks if weeks else float("nan")
    tpw_step = step_hits / weeks if weeks else float("nan")

    return {
        "window": oos_dir.parent.name,
        "ann_return": _as_float(metrics.get("ann_return")),
        "ann_vol": _as_float(metrics.get("ann_vol")),
        "sharpe": _as_float(metrics.get("sharpe")),
        "max_drawdown": _as_float(metrics.get("max_drawdown")),
        "tpw_accum": float(tpw_accum),
        "tpw_step": float(tpw_step),
        "accum_hits": int(accum_hits),
        "step_hits": int(step_hits),
        "total_dweight": total_dweight,
    }


def summarize_walkforward(
    root: Path,
    threshold: float,
    bars_per_week: float | None,
    min_weeks: float,
    mode: str,
    compare: bool,
    gate_by: str,
    gate_sharpe_min: float,
    gate_maxdd_max: float,
    gate_tpw_max: float,
) -> None:
    rows: list[dict[str, Any]] = []
    skipped = 0

    for wf_dir in sorted(root.glob("wf_*")):
        oos_dir = wf_dir / "oos"
        result = _collect(oos_dir, threshold, bars_per_week, min_weeks)
        if result is None:
            skipped += 1
            continue
        rows.append(result)

    if not rows:
        if skipped and min_weeks > 0:
            print(f"All windows shorter than {min_weeks:.2f} weeks were skipped.")
        else:
            print("No OOS windows found.")
        return

    if skipped and min_weeks > 0:
        print(f"Skipped {skipped} window(s) shorter than {min_weeks:.2f} weeks.")

    gating_enabled = (compare or mode == "both") and gate_by in {"step", "accum"}
    gate_results: list[tuple[str, str]] = []

    if gating_enabled:
        for entry in rows:
            status, detail = _evaluate_gate(
                entry, gate_by, gate_sharpe_min, gate_maxdd_max, gate_tpw_max
            )
            entry["gate_status"] = status
            entry["gate_detail"] = detail
            gate_results.append((status, detail))

    ann_returns = [row["ann_return"] for row in rows]
    ann_vols = [row["ann_vol"] for row in rows]
    sharpes = [row["sharpe"] for row in rows]
    max_drawdowns = [row["max_drawdown"] for row in rows]
    tpw_accums = [row["tpw_accum"] for row in rows]
    tpw_steps = [row["tpw_step"] for row in rows]
    accum_hits_vals = [row["accum_hits"] for row in rows]
    step_hits_vals = [row["step_hits"] for row in rows]
    total_dweights = [row["total_dweight"] for row in rows]

    avg_ann_return = _nanmean(ann_returns)
    avg_ann_vol = _nanmean(ann_vols)
    avg_sharpe = _nanmean(sharpes)
    avg_max_dd_pct = _nanmean([abs(val) for val in max_drawdowns]) * 100.0
    avg_tpw_accum = _nanmean(tpw_accums)
    avg_tpw_step = _nanmean(tpw_steps)
    avg_accum_hits = _nanmean(accum_hits_vals)
    avg_step_hits = _nanmean(step_hits_vals)
    avg_total_dweight = _nanmean(total_dweights)

    gate_passes = sum(1 for status, _ in gate_results if status == "PASS") if gating_enabled else 0

    if compare:
        header_parts = [
            "window",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_dd",
            "trades/week(step)",
            "step_hits",
            "trades/week(accum)",
            "accum_hits",
            "total_dweight",
        ]
        if gating_enabled:
            header_parts += [f"gate({gate_by})", "gate_reason"]
        print(",".join(header_parts))

        for entry in rows:
            fields = [
                entry["window"],
                f"{entry['ann_return']:.4f}",
                f"{entry['ann_vol']:.4f}",
                f"{entry['sharpe']:.3f}",
                f"{entry['max_drawdown']:.3%}",
                f"{entry['tpw_step']:.1f}",
                str(entry["step_hits"]),
                f"{entry['tpw_accum']:.1f}",
                str(entry["accum_hits"]),
                f"{entry['total_dweight']:.6f}",
            ]
            if gating_enabled:
                fields += [entry.get("gate_status", ""), entry.get("gate_detail", "")]
            print(",".join(fields))

        summary = (
            "\nAverages -> ann_return: %.4f | ann_vol: %.4f | sharpe: %.3f | max_dd: %.3f%% | "
            "trades/week(step): %.1f | step_hits: %.1f | trades/week(accum): %.1f | accum_hits: %.1f | total_dweight: %.6f"
            % (
                avg_ann_return,
                avg_ann_vol,
                avg_sharpe,
                avg_max_dd_pct,
                avg_tpw_step,
                avg_step_hits,
                avg_tpw_accum,
                avg_accum_hits,
                avg_total_dweight,
            )
        )
        print(summary)
        if gating_enabled:
            print(f"Gate pass rate ({gate_by}): {gate_passes}/{len(rows)}")
        return

    if mode == "both":
        header_parts = [
            "window",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_dd",
            "trades/week_accum",
            "trades/week_step",
            "accum_hits",
            "total_dweight",
        ]
        if gating_enabled:
            header_parts += [f"gate({gate_by})", "gate_reason"]
        print(",".join(header_parts))

        for entry in rows:
            fields = [
                entry["window"],
                f"{entry['ann_return']:.4f}",
                f"{entry['ann_vol']:.4f}",
                f"{entry['sharpe']:.3f}",
                f"{entry['max_drawdown']:.3%}",
                f"{entry['tpw_accum']:.1f}",
                f"{entry['tpw_step']:.1f}",
                str(entry["accum_hits"]),
                f"{entry['total_dweight']:.6f}",
            ]
            if gating_enabled:
                fields += [entry.get("gate_status", ""), entry.get("gate_detail", "")]
            print(",".join(fields))

        summary = (
            "\nAverages -> ann_return: %.4f | ann_vol: %.4f | sharpe: %.3f | max_dd: %.3f%% | "
            "trades/week_accum: %.1f | trades/week_step: %.1f | accum_hits: %.1f | total_dweight: %.6f"
            % (
                avg_ann_return,
                avg_ann_vol,
                avg_sharpe,
                avg_max_dd_pct,
                avg_tpw_accum,
                avg_tpw_step,
                avg_accum_hits,
                avg_total_dweight,
            )
        )
        print(summary)
        if gating_enabled:
            print(f"Gate pass rate ({gate_by}): {gate_passes}/{len(rows)}")
        return

    if mode == "accum":
        print(
            "window, ann_return, ann_vol, sharpe, max_dd, trades/week(accum), accum_hits, total_dweight"
        )
        for entry in rows:
            print(
                ",".join(
                    [
                        entry["window"],
                        f"{entry['ann_return']:.4f}",
                        f"{entry['ann_vol']:.4f}",
                        f"{entry['sharpe']:.3f}",
                        f"{entry['max_drawdown']:.3%}",
                        f"{entry['tpw_accum']:.1f}",
                        str(entry["accum_hits"]),
                        f"{entry['total_dweight']:.6f}",
                    ]
                )
            )
        print(
            "\nAverages -> ann_return: %.4f | ann_vol: %.4f | sharpe: %.3f | max_dd: %.3f%% | trades/week(accum): %.1f | accum_hits: %.1f | total_dweight: %.6f"
            % (
                avg_ann_return,
                avg_ann_vol,
                avg_sharpe,
                avg_max_dd_pct,
                avg_tpw_accum,
                avg_accum_hits,
                avg_total_dweight,
            )
        )
        return

    # mode == "step"
    print(
        "window, ann_return, ann_vol, sharpe, max_dd, trades/week(step), step_hits, total_dweight"
    )
    for entry in rows:
        print(
            ",".join(
                [
                    entry["window"],
                    f"{entry['ann_return']:.4f}",
                    f"{entry['ann_vol']:.4f}",
                    f"{entry['sharpe']:.3f}",
                    f"{entry['max_drawdown']:.3%}",
                    f"{entry['tpw_step']:.1f}",
                    str(entry["step_hits"]),
                    f"{entry['total_dweight']:.6f}",
                ]
            )
        )
    print(
        "\nAverages -> ann_return: %.4f | ann_vol: %.4f | sharpe: %.3f | max_dd: %.3f%% | trades/week(step): %.1f | step_hits: %.1f | total_dweight: %.6f"
        % (
            avg_ann_return,
            avg_ann_vol,
            avg_sharpe,
            avg_max_dd_pct,
            avg_tpw_step,
            avg_step_hits,
            avg_total_dweight,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score OOS Gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )
    parser.add_argument("--wf_root", required=True)
    parser.add_argument(
        "--thr",
        type=float,
        default=0.02,
        help="threshold for counting a trade (align with dweight_threshold)",
    )
    parser.add_argument(
        "--bars_per_week",
        type=float,
        default=None,
        help="override if no meta; leave empty to auto",
    )
    parser.add_argument(
        "--min_weeks",
        type=float,
        default=0.5,
        help="skip windows shorter than this span (weeks); set 0 to keep all",
    )
    parser.add_argument(
        "--count_mode",
        choices=["step", "accum", "both"],
        default="step",
        help="report trade counts per step, per accumulated threshold, or both",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="print step/accum metrics side by side for the same windows",
    )
    parser.add_argument(
        "--gate_by",
        choices=["step", "accum"],
        default="accum",
        help="which trades/week metric to use when computing gate status (only active for both/compare modes)",
    )
    parser.add_argument(
        "--gate_sharpe_min",
        type=float,
        default=0.0,
        help="minimum Sharpe ratio required to pass the gate",
    )
    parser.add_argument(
        "--gate_maxdd_max",
        type=float,
        default=0.20,
        help="maximum absolute max drawdown allowed to pass the gate (as fraction, e.g. 0.20 = 20%%)",
    )
    parser.add_argument(
        "--gate_tpw_max",
        type=float,
        default=20.0,
        help="maximum trades per week allowed to pass the gate",
    )
    args = parser.parse_args()

    summarize_walkforward(
        Path(args.wf_root),
        threshold=args.thr,
        bars_per_week=args.bars_per_week,
        min_weeks=args.min_weeks,
        mode=args.count_mode,
        compare=args.compare,
        gate_by=args.gate_by,
        gate_sharpe_min=args.gate_sharpe_min,
        gate_maxdd_max=args.gate_maxdd_max,
        gate_tpw_max=args.gate_tpw_max,
    )


if __name__ == "__main__":
    main()
