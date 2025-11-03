from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


def write_markdown(out: Path, metrics: Dict[str, Any], symbols: Iterable[str]) -> None:
    lines = [
        "# RL3  Training Report",
        "## Summary",
        f"- Annualized Return: {metrics['ann_return']:.4f}",
        f"- Annualized Vol: {metrics['ann_vol']:.4f}",
        f"- Sharpe: {metrics['sharpe']:.3f}",
        f"- Max Drawdown: {metrics['max_drawdown']:.3%}",
        f"- Symbols: {', '.join(symbols)}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
