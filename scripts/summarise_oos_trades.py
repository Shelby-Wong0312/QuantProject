#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarise OOS trade counts per week using recorded weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from src.quantproject.data_pipeline.loaders.bars import load_and_align_router


def _infer_config_path(oos_dir: Path) -> Optional[Path]:
    candidates: Iterable[Path] = (
        oos_dir.parent / "train" / "config.json",
        oos_dir.parent / "config.json",
        oos_dir.parent.parent / "config.json",
    )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_oos(oos_dir: Path, tolerance: float) -> dict:
    if not oos_dir.exists():
        raise FileNotFoundError(f"OOS directory not found: {oos_dir}")

    meta = json.loads((oos_dir / "oos_meta.json").read_text(encoding="utf-8"))
    weights = pd.DataFrame(
        json.loads((oos_dir / "oos_weights.json").read_text(encoding="utf-8")),
        columns=meta.get("symbols", []),
    )
    returns = pd.Series(json.loads((oos_dir / "oos_returns.json").read_text(encoding="utf-8")))

    return {
        "meta": meta,
        "weights": weights,
        "returns": returns,
        "tolerance": tolerance,
    }


def _prepare_index(weights: pd.DataFrame, meta: dict, config: dict) -> pd.DataFrame:
    symbols = config["symbols"]
    timeframe = config.get("timeframe", meta.get("timeframe", "60m"))
    start = meta["start"]
    end = meta["end"]

    prices = load_and_align_router(symbols, start, end, timeframe)
    if not prices:
        raise ValueError(f"No price data fetched for OOS window {start} -> {end}")

    index = next(iter(prices.values())).index[: len(weights)]
    weights = weights.copy()
    weights.index = index
    return weights


def summarise_trades(oos_dir: Path, tolerance: float = 1e-6) -> dict:
    payload = _load_oos(oos_dir, tolerance)
    meta = payload["meta"]
    cfg_path = _infer_config_path(oos_dir)
    if not cfg_path:
        raise FileNotFoundError(f"Config file not found for {oos_dir}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    weights = _prepare_index(payload["weights"], meta, config)
    delta = weights.diff().abs().sum(axis=1)
    trade_mask = delta > tolerance

    weekly_counts = trade_mask.astype(int).resample("W").sum()
    total_trades = int(trade_mask.sum())
    weeks = len(weekly_counts)
    avg_trades_per_week = float(total_trades / weeks) if weeks else float("nan")

    return {
        "oos_dir": str(oos_dir),
        "start": meta["start"],
        "end": meta["end"],
        "tolerance": tolerance,
        "total_steps": len(weights),
        "total_trades": total_trades,
        "weeks": weeks,
        "avg_trades_per_week": avg_trades_per_week,
        "weekly_counts": weekly_counts.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise OOS trade counts per week")
    parser.add_argument("paths", nargs="+", help="One or more OOS directories")
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Threshold on |Δw| to count a trade"
    )
    args = parser.parse_args()

    for path_str in args.paths:
        summary = summarise_trades(Path(path_str), tolerance=args.tolerance)
        print("==", summary["oos_dir"])
        print(f"Window : {summary['start']} -> {summary['end']}")
        print(f"Steps  : {summary['total_steps']}")
        print(f"Trades : {summary['total_trades']}")
        print(f"Weeks  : {summary['weeks']}")
        print(f"Avg trades/week : {summary['avg_trades_per_week']:.2f}")
        print("Weekly counts   :", summary["weekly_counts"])
        print()


if __name__ == "__main__":
    main()
