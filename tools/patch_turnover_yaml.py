from pathlib import Path

import yaml

paths = [
    Path("configs/train/ppo_mixed_60m_lstm_cost_high.yaml"),
    Path("configs/train/ppo_mixed_60m_lstm_cost_low.yaml"),
]

for path in paths:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    action = data.setdefault("action", {})
    action["max_dweight"] = 0.006
    action["dweight_threshold"] = 0.02

    reward = data.setdefault("reward", {})
    reward["lambda_turnover"] = 0.004

    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"patched: {path}")
