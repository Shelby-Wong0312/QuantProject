from pathlib import Path
import yaml


def test_train_yaml_loads():
    p = Path("configs/train/ppo_baseline.yaml")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert "symbols" in cfg and len(cfg["symbols"]) >= 2
    assert "obs" in cfg and "features" in cfg["obs"]
    train = cfg.get("train", {})
    assert train.get("policy", "MlpLstmPolicy") == "MlpLstmPolicy"
    assert "policy_kwargs" in train
