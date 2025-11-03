import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
py = ROOT / "pyproject.toml"

BASE = """[tool.black]
line-length = 100
extend-exclude = '(venv|\\.venv|build|dist|__pycache__|\\.git)'

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E","F","W","I"]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**.py" = ["E402"]
"""


def write(p: Path, s: str):
    p.write_text(s, encoding="utf-8", newline="\\n")


if py.exists():
    raw = py.read_text(encoding="utf-8", errors="ignore")
    if raw.count("[tool.black]") > 1:
        shutil.copy(py, py.with_suffix(".toml.bak"))
        # 砍掉所有 [tool.black] 區塊，最後補上標準設定
        pat = re.compile(r"(?ms)^\\[tool\\.black\\][^\\[]*")
        cleaned = pat.sub("", raw).strip()
        write(py, (cleaned + "\n\n" + BASE).strip() + "\n")
    else:
        # 沒有就補，有就保留原有內容並追加缺的段
        need = []
        if "[tool.black]" not in raw:
            need.append("black")
        if "[tool.ruff]" not in raw:
            need.append("ruff")
        if need:
            write(py, raw.strip() + "\n\n" + BASE)
else:
    write(py, BASE)

print("[pyproject] fixed")
