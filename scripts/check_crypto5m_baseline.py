"""
檢查 RL3 Crypto 5m Walk-Forward baseline 視窗摘要。

使用：
    python scripts/check_crypto5m_baseline.py
"""

from pathlib import Path
import json
import math

# 直接指定專案根（用 / 避免反斜線問題）
PROJ = Path("C:/Users/niuji/Documents/QuantProject")
WF_ROOT = PROJ / "runs" / "rl3" / "walkforward"


def _safe_load_json(p: Path):
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NaN"
    return f"{x*100:.3f}%%"


def main():
    if not WF_ROOT.exists():
        print(f"[FATAL] WF_ROOT not found: {WF_ROOT}")
        return

    wfs = sorted(p for p in WF_ROOT.glob("wf_*") if (p / "oos").exists())
    if not wfs:
        print(f"[WARN] no wf_*/oos found under {WF_ROOT}")
        return

    print(f"[INFO] Scanning {len(wfs)} window(s) under {WF_ROOT}")
    print("window, calendar, start, end, sharpe, max_dd, tpw_step, tpw_accum, weeks")

    for wf in wfs:
        oos = wf / "oos"
        meta = _safe_load_json(oos / "oos_meta.json")
        mtr  = _safe_load_json(oos / "oos_metrics.json")

        start = meta.get("start") or meta.get("oos_start") or "?"
        end   = meta.get("end")   or meta.get("oos_end")   or "?"
        cal   = mtr.get("calendar", "crypto")

        sharpe = mtr.get("sharpe")
        max_dd = mtr.get("max_drawdown")
        tpw_s  = mtr.get("trades_week_step")   or mtr.get("tpw_step")
        tpw_a  = mtr.get("trades_week_accum")  or mtr.get("tpw_accum")
        weeks  = mtr.get("weeks")

        def _fmt(x):
            if x is None:
                return "NaN"
            try:
                if isinstance(x, str):
                    return x
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    return "NaN"
                return f"{x:.3f}"
            except Exception:
                return str(x)

        print(
            f"{wf.name},{cal},{start},{end},"
            f"{_fmt(sharpe)},{_fmt_pct(max_dd)},"
            f"{_fmt(tpw_s)},{_fmt(tpw_a)},{_fmt(weeks)}"
        )

    print("\n[NOTE] 這只是 summarize，不做 Gate 判斷；Gate 還是用 score_oos_gate.py。")


if __name__ == "__main__":
    main()
