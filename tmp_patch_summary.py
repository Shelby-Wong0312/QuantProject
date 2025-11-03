from pathlib import Path

path = Path("rl3/eval/rollout.py")
text = path.read_text(encoding="utf-8")

# Insert episode_resets initialization after episode_starts definition
old = "    lstm_states: Optional[Tuple[np.ndarray, ...]] = None\n    episode_starts = np.array([True], dtype=bool)\n\n    if is_recurrent and hasattr(policy, 'initial_state'):\n"
new = "    lstm_states: Optional[Tuple[np.ndarray, ...]] = None\n    episode_starts = np.array([True], dtype=bool)\n    episode_resets = 1\n\n    if is_recurrent and hasattr(policy, 'initial_state'):\n"
if old not in text:
    raise SystemExit("anchor for episode_starts not found")
text = text.replace(old, new, 1)

# After while loop, before metrics computation, add stats for weights
old_loop_tail = "        if weight_vec is not None:\n            weights.append([float(x) for x in weight_vec])\n\n    metrics = compute(\n"
if old_loop_tail not in text:
    raise SystemExit("loop tail anchor not found")
addition = "\n    weights_arr = np.asarray(weights, dtype=float) if weights else np.empty((0, 0), dtype=float)\n    if weights_arr.size == 0:\n        unique_weights = 0\n        total_dweight = 0.0\n    else:\n        rounded = np.round(weights_arr, 10)\n        try:\n            unique_weights = int(np.unique(rounded, axis=0).shape[0])\n        except ValueError:\n            unique_weights = int(len({tuple(row) for row in rounded.tolist()}))\n        total_dweight = float(np.abs(np.diff(weights_arr, axis=0)).sum()) if len(weights_arr) > 1 else 0.0\n\n    metrics = compute(\n"
text = text.replace(old_loop_tail, addition, 1)

# After metrics computed, inject our stats into metrics dict before writing
old_metrics = '    metrics = compute(\n        returns=returns,\n        equity=equity,\n        timeframe=cfg.get("timeframe", "5min"),\n        symbols=cfg.get("symbols", []),\n    )\n\n    out_dir.mkdir(parents=True, exist_ok=True)\n'
new_metrics = '    metrics = compute(\n        returns=returns,\n        equity=equity,\n        timeframe=cfg.get("timeframe", "5min"),\n        symbols=cfg.get("symbols", []),\n    )\n    metrics["unique_weights"] = int(unique_weights)\n    metrics["total_dweight"] = float(total_dweight)\n    metrics["episode_resets"] = int(episode_resets)\n\n    out_dir.mkdir(parents=True, exist_ok=True)\n'
text = text.replace(old_metrics, new_metrics, 1)

path.write_text(text, encoding="utf-8")
