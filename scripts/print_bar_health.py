from __future__ import annotations

# add repo/src to sys.path so "src.quantproject..." works when running this script
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathlib import Path
from pprint import pprint

import yaml

from src.quantproject.data_pipeline.loaders.bars import load_and_align


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    symbols = cfg["symbols"]
    start = cfg["start"]
    end = cfg["end"]
    timeframe = cfg.get("timeframe", "5min")
    try:
        data = load_and_align(symbols, start, end, timeframe)
    except ValueError as exc:
        print(f"Failed to load data: {exc}")
        return

    missing = [s for s in symbols if s not in data]
    if missing:
        print(f"Skipped symbols (no data): {', '.join(missing)}")

    pprint({symbol: df.shape for symbol, df in data.items()})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/print_bar_health.py <config.yaml>")
    main(sys.argv[1])
