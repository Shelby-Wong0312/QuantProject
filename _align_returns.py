import json
from pathlib import Path
import pandas as pd
from src.quantproject.data_pipeline.loaders.bars import load_and_align_router
from eval.metrics import compute

base = Path("runs/rl3/wf_mixed_60m_strict")
for wf_dir in sorted([d for d in base.glob("wf_*") if d.is_dir()]):
    result = json.loads((wf_dir / "result.json").read_text())
    oos_dir = Path(result["oos_dir"])
    json.loads((Path(result["train_dir"]) / "config.json").read_text())["symbols"]

    returns_logged = json.loads((oos_dir / "oos_returns.json").read_text())
    weights = pd.DataFrame(json.loads((oos_dir / "oos_weights.json").read_text()), columns=symbols)
    start = result["test_start"]
    end = result["test_end"]
    prices = load_and_align_router(symbols, start, end, "60m")
    rets = pd.DataFrame({s: df["close"].pct_change().fillna(0.0) for s, df in prices.items()})
    rets = rets.iloc[: len(weights)]
    weights.index = rets.index[: len(weights)]
    recomputed = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    diff = recomputed.values - returns_logged[: len(recomputed)]
    print(wf_dir.name, "max_abs_diff", round(abs(diff).max(), 6))
