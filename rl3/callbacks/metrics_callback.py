from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    def __init__(self, log_dir: Path, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        base = getattr(self, "training_env", None)
        if base is None:
            return

        # unwrap VecEnv/VecNormalize/VecCheckNan wrappers
        visited = set()
        while hasattr(base, "venv") and base not in visited:
            visited.add(base)
            base = base.venv

        if hasattr(base, "envs") and base.envs:
            base = base.envs[0]

        while hasattr(base, "env"):
            base = base.env

        if hasattr(base, "get_training_artifacts"):
            artifacts = base.get_training_artifacts()
            self._write_artifacts(artifacts)

    def _write_artifacts(self, artifacts: Dict[str, Any]) -> None:
        import json

        from eval.metrics import compute
        from eval.reports import write_markdown

        self.log_dir.mkdir(parents=True, exist_ok=True)
        returns_path = self.log_dir / "returns.json"
        equity_path = self.log_dir / "equity.json"
        weights_path = self.log_dir / "weights.json"

        returns_path.write_text(json.dumps(list(artifacts.get("returns", []))), encoding="utf-8")
        equity_path.write_text(json.dumps(list(artifacts.get("equity", []))), encoding="utf-8")
        weights_path.write_text(json.dumps(list(artifacts.get("weights", []))), encoding="utf-8")

        metrics = compute(list(artifacts.get("returns", [])), list(artifacts.get("equity", [])))

        symbols: List[str] = []
        cfg_path = self.log_dir / "config.json"
        if cfg_path.exists():
            try:
                cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                symbols = list(cfg_data.get("symbols", []))
            except Exception:
                symbols = []

        write_markdown(self.log_dir / "report.md", metrics, symbols or [])