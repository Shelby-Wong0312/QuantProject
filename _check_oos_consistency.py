import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _parse_iso(value: str) -> datetime:
    value = str(value).strip()
    if not value:
        raise ValueError("empty iso timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return datetime.fromisoformat(f"{value}T00:00:00+00:00")


def _weeks_from_meta(oos_dir: Path) -> float | None:
    meta_path = oos_dir / "oos_meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    start = payload.get("start")
    end = payload.get("end")
    if not start or not end:
        return None
    try:
        start_dt = _parse_iso(start)
        end_dt = _parse_iso(end)
    except Exception:
        return None
    seconds = max(0.0, (end_dt - start_dt).total_seconds())
    return max(1e-9, seconds / (7 * 24 * 3600.0))


def _load_weights_df(oos_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = oos_dir / "weights.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)

    json_path = oos_dir / "oos_weights.json"
    if not json_path.exists():
        return None

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and {"index", "columns", "data"} <= set(
        payload.keys()
    ):
        return pd.DataFrame(
            payload["data"], index=payload["index"], columns=payload["columns"]
        )
    if isinstance(payload, list) and payload:
        sample = payload[0]
        if isinstance(sample, dict) and "weights" in sample:
            index: List[Optional[str]] = []
            rows: List[List[float]] = []
            for record in payload:
                index.append(
                    record.get("ts") or record.get("t") or record.get("timestamp")
                )
                rows.append(record["weights"])
            return pd.DataFrame(rows, index=index).sort_index()
        if isinstance(sample, (list, tuple)):
            return pd.DataFrame(payload)
    if isinstance(payload, (list, tuple)):
        return pd.DataFrame(np.asarray(payload, dtype=float))
    return None


def _count_trades(diff: pd.DataFrame, threshold: float) -> tuple[int, int]:
    if diff.empty:
        return 0, 0
    arr = np.nan_to_num(diff.to_numpy(dtype=float), nan=0.0)
    step_hits = int((arr >= threshold).any(axis=1).sum())
    accum_vec = np.zeros(arr.shape[1], dtype=float)
    accum_hits = 0
    for row in arr:
        accum_vec += row
        mask = accum_vec >= threshold
        if mask.any():
            accum_hits += 1
            accum_vec[mask] = 0.0
    return step_hits, accum_hits


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OOS weight consistency")
    parser.add_argument("--wf_root", required=True)
    parser.add_argument("--thr", type=float, required=True)
    parser.add_argument("--bars_per_week", type=float, default=32.5)
    parser.add_argument("--min_weeks", type=float, default=0.5)
    parser.add_argument(
        "--count_mode",
        choices=["step", "accum", "both"],
        default="step",
        help="step: per-bar threshold; accum: accumulate until threshold; both: report both",
    )
    args = parser.parse_args()

    root = Path(args.wf_root)
    rows = []
    skipped = 0

    for wf_dir in sorted(root.glob("wf_*")):
        oos_dir = wf_dir / "oos"
        weights = _load_weights_df(oos_dir)
        if weights is None or weights.empty:
            continue

        diff = weights.diff().abs().fillna(0.0)
        total_rows = len(weights)
        unique_rows = int(weights.drop_duplicates().shape[0])
        total_dweight = float(np.nansum(diff))

        step_hits, accum_hits = _count_trades(diff, args.thr)
        meta_weeks = _weeks_from_meta(oos_dir) or 0.0
        bars_weeks = (
            (total_rows / args.bars_per_week)
            if args.bars_per_week and args.bars_per_week > 0
            else 0.0
        )
        denom_weeks = max(meta_weeks, bars_weeks, 1e-9)
        if denom_weeks < args.min_weeks:
            skipped += 1
            continue

        rows.append(
            {
                "window": wf_dir.name,
                "rows": total_rows,
                "unique_rows": unique_rows,
                "total_dweight": total_dweight,
                "step_hits": step_hits,
                "accum_hits": accum_hits,
                "step_tpweek": step_hits / denom_weeks,
                "accum_tpweek": accum_hits / denom_weeks,
            }
        )

    if not rows:
        if skipped and args.min_weeks > 0:
            print(f"All windows shorter than {args.min_weeks:.2f} weeks were skipped.")
        else:
            print(f"No OOS windows found under {root}")
        return

    if skipped and args.min_weeks > 0:
        print(f"Skipped {skipped} window(s) shorter than {args.min_weeks:.2f} weeks.")

    if args.count_mode == "both":
        header = "window, rows, unique_rows, trades/week_accum, trades/week_step, accum_hits, total_dweight"
    elif args.count_mode == "accum":
        header = (
            "window, rows, unique_rows, total_dweight, accum_hits, trades/week(accum)"
        )
    else:
        header = (
            "window, rows, unique_rows, total_dweight, step_hits, trades/week(step)"
        )
    print(header)

    for entry in sorted(rows, key=lambda item: item["window"]):
        base = [entry["window"], str(entry["rows"]), str(entry["unique_rows"])]
        if args.count_mode == "both":
            fields = base + [
                f"{entry['accum_tpweek']:.3f}",
                f"{entry['step_tpweek']:.3f}",
                str(entry["accum_hits"]),
                f"{entry['total_dweight']:.6f}",
            ]
        elif args.count_mode == "accum":
            fields = base + [
                f"{entry['total_dweight']:.6f}",
                str(entry["accum_hits"]),
                f"{entry['accum_tpweek']:.3f}",
            ]
        else:
            fields = base + [
                f"{entry['total_dweight']:.6f}",
                str(entry["step_hits"]),
                f"{entry['step_tpweek']:.3f}",
            ]
        print(",".join(fields))

    row_counts = [row["rows"] for row in rows]
    unique_counts = [row["unique_rows"] for row in rows]
    total_dweights = [row["total_dweight"] for row in rows]
    step_hits_vals = [row["step_hits"] for row in rows]
    accum_hits_vals = [row["accum_hits"] for row in rows]
    step_tpweeks = [row["step_tpweek"] for row in rows]
    accum_tpweeks = [row["accum_tpweek"] for row in rows]

    avg_rows = _nanmean(row_counts)
    avg_unique_rows = _nanmean(unique_counts)
    avg_total_dweight = _nanmean(total_dweights)
    avg_step_hits = _nanmean(step_hits_vals)
    avg_accum_hits = _nanmean(accum_hits_vals)
    avg_step_tpweek = _nanmean(step_tpweeks)
    avg_accum_tpweek = _nanmean(accum_tpweeks)

    if args.count_mode == "both":
        print(
            "\nAverages -> rows: %.1f | unique_rows: %.1f | trades/week_accum: %.3f | trades/week_step: %.3f | accum_hits: %.1f | total_dweight: %.6f"
            % (
                avg_rows,
                avg_unique_rows,
                avg_accum_tpweek,
                avg_step_tpweek,
                avg_accum_hits,
                avg_total_dweight,
            )
        )
    elif args.count_mode == "accum":
        print(
            "\nAverages -> rows: %.1f | unique_rows: %.1f | total_dweight: %.6f | accum_hits: %.1f | trades/week(accum): %.3f"
            % (
                avg_rows,
                avg_unique_rows,
                avg_total_dweight,
                avg_accum_hits,
                avg_accum_tpweek,
            )
        )
    else:
        print(
            "\nAverages -> rows: %.1f | unique_rows: %.1f | total_dweight: %.6f | step_hits: %.1f | trades/week(step): %.3f"
            % (
                avg_rows,
                avg_unique_rows,
                avg_total_dweight,
                avg_step_hits,
                avg_step_tpweek,
            )
        )


if __name__ == "__main__":
    main()
