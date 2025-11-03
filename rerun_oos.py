from pathlib import Path
from rl3.eval.rollout import run_oos_eval

RUN_ROOT = Path("runs/rl3/wf_mixed_60m_lstm_tband030/wf_02")
run_oos_eval(
    model_path=RUN_ROOT / "train/model.zip",
    vecnorm_path=RUN_ROOT / "train/vecnormalize.pkl",
    cfg_path=RUN_ROOT / "train/config.json",
    start="2025-09-18",
    end="2025-09-24",
    out_dir=RUN_ROOT / "oos",
    deterministic=True,
)
