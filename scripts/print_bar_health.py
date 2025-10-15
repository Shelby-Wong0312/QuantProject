from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.quantproject.data_pipeline.loaders.bars import load_and_align_router


def _fetch(symbols, start, end, timeframe, label: str) -> None:
    data = load_and_align_router(symbols, start, end, timeframe)
    if not data:
        print(f"[{label}] no data returned; check router configuration or inputs.")
        return
    shapes = {symbol: df.shape for symbol, df in data.items()}
    print(f"[{label}] {shapes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect data availability and optionally warm caches for a training config.",
    )
    parser.add_argument("config", help="Path to training YAML config")
    parser.add_argument(
        "--include-oos",
        action="store_true",
        help="If the config includes an 'oos' section (dict or list), fetch those windows as well.",
    )
    parser.add_argument(
        "--oos-start",
        help="Optional override for OOS start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--oos-end",
        help="Optional override for OOS end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--oos-timeframe",
        help="Optional timeframe override for extra OOS range (defaults to config timeframe).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    symbols = cfg["symbols"]
    timeframe = cfg.get("timeframe", "5min")

    _fetch(symbols, cfg["start"], cfg["end"], timeframe, "train")

    extra_ranges: list[dict] = []
    if args.include_oos:
        oos_block = cfg.get("oos")
        if isinstance(oos_block, dict):
            extra_ranges.append(oos_block)
        elif isinstance(oos_block, list):
            extra_ranges.extend(item for item in oos_block if isinstance(item, dict))
    if args.oos_start and args.oos_end:
        extra_ranges.append(
            {
                "start": args.oos_start,
                "end": args.oos_end,
                "timeframe": args.oos_timeframe or timeframe,
            }
        )

    for idx, item in enumerate(extra_ranges, 1):
        start = item.get("start")
        end = item.get("end")
        if not (start and end):
            continue
        tf = item.get("timeframe", timeframe)
        _fetch(symbols, start, end, tf, f"oos#{idx}")


if __name__ == "__main__":
    main()
