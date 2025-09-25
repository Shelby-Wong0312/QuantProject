from __future__ import annotations

import json
import math
from pathlib import Path


def test_report_has_real_values() -> None:
    p_ret = Path("runs/rl3/baseline/returns.json")
    p_eq = Path("runs/rl3/baseline/equity.json")
    p_rep = Path("runs/rl3/baseline/report.md")

    if not (p_ret.exists() and p_eq.exists() and p_rep.exists()):
        return

    returns = json.loads(p_ret.read_text(encoding="utf-8"))
    equity = json.loads(p_eq.read_text(encoding="utf-8"))
    text = p_rep.read_text(encoding="utf-8")

    assert any(abs(x) > 1e-12 for x in returns), "returns should not be all zeros"
    assert not any(math.isnan(x) for x in returns), "returns should not contain NaN"
    assert any(abs(equity[i] - equity[i - 1]) > 1e-8 for i in range(1, len(equity))), "equity should move"
    assert "Sharpe:" in text and "Max Drawdown:" in text
