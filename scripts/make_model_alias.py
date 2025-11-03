#!/usr/bin/env python
from __future__ import annotations

import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/rl3/baseline")
src = run_dir / "ppo_model.zip"
dst = run_dir / "model.zip"

if src.exists():
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"Aliased {src} -> {dst}")
    else:
        print("model.zip already exists; nothing to do.")
else:
    print("ppo_model.zip not found; nothing to alias.")
