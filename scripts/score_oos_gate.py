from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timezone

def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _parse_iso(s: str) -> datetime:
    # accept 'YYYY-MM-DD' or full ISO
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.fromisoformat(s + "T00:00:00+00:00")

def _weeks_from_meta(oos_dir: Path) -> float | None:
    meta = oos_dir / "oos_meta.json"
    if not meta.exists():
        return None
    m = _load_json(meta)
    start = _parse_iso(str(m.get("start")))
    end   = _parse_iso(str(m.get("end")))
    seconds = max(0.0, (end - start).total_seconds())
    return max(1e-9, seconds / (7 * 24 * 3600.0))  # avoid div by zero

def _count_trades(weights: np.ndarray, thr: float) -> int:
    if len(weights) < 2:
        return 0
    dW = np.abs(np.diff(weights, axis=0))
    return int((dW >= thr).any(axis=1).sum())

def _collect(oos_dir: Path, thr: float, bars_per_week: float | None) -> tuple[str, float, float, float, float, float]:
    m = _load_json(oos_dir / "oos_metrics.json")
    w = np.array(_load_json(oos_dir / "oos_weights.json"), dtype=float)
    trades = _count_trades(w, thr=thr)

    weeks = _weeks_from_meta(oos_dir)
    if weeks is None:
        # fallback: infer by bar count
        bpw = bars_per_week if bars_per_week else 168.0
        weeks = max(1e-9, len(w) / bpw)

    tpw = trades / weeks
    return (oos_dir.parent.name, m["ann_return"], m["ann_vol"], m["sharpe"], m["max_drawdown"], tpw)

def summarize_walkforward(root: Path, thr: float, bars_per_week: float | None) -> None:
    rows: List[tuple] = []
    for wf in sorted(root.glob("wf_*")):
        oos = wf / "oos"
        if not (oos / "oos_metrics.json").exists():
            continue
        rows.append(_collect(oos, thr=thr, bars_per_week=bars_per_week))
    if not rows:
        print("No OOS windows found.")
        return

    print("window, ann_return, ann_vol, sharpe, max_dd, trades/week")
    for r in rows:
        print(",".join([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.3f}", f"{r[4]:.3%}", f"{r[5]:.1f}"]))

    s = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    avg = s.mean(axis=0)
    print("\nAverages -> ann_return: %.4f | ann_vol: %.4f | sharpe: %.3f | max_dd: %.3f%% | trades/week: %.1f"
          % (avg[0], avg[1], avg[2], avg[3]*100, avg[4]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf_root", required=True)
    ap.add_argument("--thr", type=float, default=0.02, help="threshold for counting a trade (align with dweight_threshold)")
    ap.add_argument("--bars_per_week", type=float, default=None, help="override if no meta; leave empty to auto")
    args = ap.parse_args()
    summarize_walkforward(Path(args.wf_root), thr=args.thr, bars_per_week=args.bars_per_week)
