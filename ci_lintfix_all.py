import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {"venv", ".venv", "build", "dist", ".git", "__pycache__"}

def should_visit(p: Path) -> bool:
    parts = set(p.parts)
    return not (parts & EXCLUDE_DIRS)

def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        for enc in ("utf-8-sig","cp950","big5","latin1"):
            try:
                return p.read_text(encoding=enc)
            except Exception:
                pass
        return p.read_bytes().decode("utf-8", errors="replace")

def write(p: Path, s: str):
    p.write_text(s, encoding="utf-8", newline="\n")

def fix_constant_fstrings(code: str) -> str:
    # f"沒有{}" -> ""
    code = re.sub(r'f"([^{"}\n]*)"', r'"\1"', code)
    code = re.sub(r"f'([^{'}\n]*)'", r"'\1'", code)
    return code

def drop_unused_assigns(code: str) -> str:
    # 把特定 unused 變數的賦值移除，保留右側表達式（含 await）
    pat = re.compile(r'^(\s*)(order_id|data|report|best_params|symbols|graph|obs|start_cpu|signals)\s*=\s*(await\s+)?(.+)$', re.MULTILINE)
    def _rep(m):
        indent, _, awt, rhs = m.groups()
        awt = awt or ""
        return f"{indent}{awt}{rhs}"
    return pat.sub(_rep, code)

def fix_bare_except(code: str) -> str:
    return re.sub(r'^(\s*)except\s*:\s*$', r'\1except Exception:', code, flags=re.MULTILINE)

def normalize_logger_weird_plus(code: str, path: Path) -> str:
    # 通殺 "" + 文字
    code = re.sub(r'logger\.(info|warning|error)\(\s*""\s*\+\s*"([^"]*)"\s*\)', r'logger.\1(" \2")', code)
    # 沒加第二個引號的各種迷因，特例修正：
    code = code.replace('logger.info(" Authentication successful!")', 'logger.info(" Authentication successful!")')
    code = code.replace('logger.info(" Authentication successful!")', 'logger.info(" Authentication successful!")')
    # config：API Key loaded
    if "config.py" in str(path).replace("\\", "/"):
        code = re.sub(
            r'logger\.info\(\s*""\s*\+\s*API\s*Key\s*loaded[^\)]*\)',
            'logger.info(f" API Key loaded: {CAPITAL_API_KEY[:4]}...{CAPITAL_API_KEY[-4:]}")',
            code, flags=re.IGNORECASE
        )
    # .env found
    code = re.sub(
        r'logger\.info\(\s*""\s*\+\s*Found[^\)]*\)',
        'logger.info(" Found .env file at: %s", env_file_path)',
        code, flags=re.IGNORECASE
    )
    # 通用：少了引號的警告/錯誤
    code = code.replace('logger.warning(".env file not found in project directory!")', 'logger.warning(".env file not found in project directory!")')
    code = code.replace('logger.error("API credentials not found!")', 'logger.error("API credentials not found!")')
    return code

def patch_files():
    py_files = [p for p in ROOT.rglob("*.py") if should_visit(p)]
    for p in py_files:
        s0 = read(p)
        s = s0
        s = normalize_logger_weird_plus(s, p)
        s = fix_constant_fstrings(s)
        s = drop_unused_assigns(s)
        s = fix_bare_except(s)
        if s != s0:
            write(p, s)

def ensure_pyproject():
    pj = ROOT / "pyproject.toml"
    base = """[tool.black]
line-length = 100
extend-exclude = '(venv|\\.venv|build|dist|__pycache__|\\.git)'

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E","F","W","I"]
ignore = []
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**.py" = ["E402"]
"""
    if pj.exists():
        text = read(pj)
        # 簡單合併：補上 per-file-ignores 與 black 排除
        if "[tool.black]" not in text or "extend-exclude" not in text:
            text += "\n" + base.split("\n\n",1)[0] + "\n"
        if "[tool.ruff.lint.per-file-ignores]" not in text:
            text += "\n" + base.split("\n\n",2)[2] + "\n"
        write(pj, text)
    else:
        write(pj, base)

if __name__ == "__main__":
    patch_files()
    ensure_pyproject()
    print("[all-fix] code patched & pyproject ensured")
