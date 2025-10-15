from pathlib import Path

path = Path("scripts/walkforward_train_eval.py")
with path.open("r", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.strip().startswith("def _ensure_min_train_span("):
        start = i
        break
else:
    raise SystemExit("function not found")
for j in range(start, len(lines)):
    if lines[j].startswith(
        "    return train_start, extended_end.date().isoformat(), True"
    ):
        end = j + 1
        break
else:
    raise SystemExit("return line not found")
block = lines[start:end]
for idx, line in enumerate(block):
    line_num = start + idx + 1
    print(f"{line_num:03}: {line.rstrip()}")
