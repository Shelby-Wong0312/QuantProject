import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        for enc in ("utf-8-sig", "cp950", "big5", "latin1"):
            try:
                return p.read_text(encoding=enc)
            except Exception:
                pass
        return p.read_bytes().decode("utf-8", errors="replace")


def write_text_utf8(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def ensure_import(path: Path, import_line: str):
    txt = read_text_safe(path)
    if import_line not in txt:
        first_non_comment = 0
        lines = txt.splitlines()
        # 找第一個不是 shebang/空行/註解的位置
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("#!"):
                continue
            if line.strip().startswith(("#", '"""', "'''")) or line.strip() == "":
                continue
            first_non_comment = i
            break
        lines.insert(first_non_comment, import_line)
        write_text_utf8(path, "\n".join(lines))


def replace_line_contains(path: Path, needle_sub_pairs):
    txt = read_text_safe(path)
    for needle, new_line in needle_sub_pairs:
        lines = txt.splitlines()
        changed = False
        for i, line in enumerate(lines):
            if needle in line:
                lines[i] = new_line
                changed = True
        if changed:
            txt = "\n".join(lines)
    write_text_utf8(path, txt)


def regex_sub_file(path: Path, pattern: str, repl: str):
    txt = read_text_safe(path)
    new = re.sub(pattern, repl, txt, flags=re.MULTILINE)
    if new != txt:
        write_text_utf8(path, new)


def ensure_header_def(path: Path, var_name: str, default_value: str = "None"):
    txt = read_text_safe(path)
    if re.search(rf"^[ \t]*{re.escape(var_name)}[ \t]*=", txt, flags=re.MULTILINE):
        return
    # 插在檔案開頭的 import 之後、第一個空行之後
    lines = txt.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:50]):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, f"{var_name} = {default_value}")
    write_text_utf8(path, "\n".join(lines))


# 1) 修 logger 爛字串
cfg = ROOT / "config" / "config.py"
if cfg.exists():
    replace_line_contains(
        cfg,
        [
            (
                "logger.warning(",
                '    logger.warning(".env file not found in project directory!")',
            ),
            (
                'logger.info("" +',
                '    logger.info(" Found .env file at: %s", env_file_path)',
            ),
        ],
    )

cap = ROOT / "src" / "connectors" / "capital_com_api.py"
if cap.exists():
    replace_line_contains(
        cap,
        [
            (
                'logger.info(" Authentication successful!")',
                '        logger.info(" Authentication successful!")',
            ),
            (
                "logger.info( Identifier:",
                '        logger.info("Identifier: %s", api.identifier)',
            ),
        ],
    )

# 2) providers/__init__.py 把 \n 文字修回真正換行
prov = ROOT / "src" / "quantproject" / "data_pipeline" / "providers" / "__init__.py"
if prov.exists():
    raw = read_text_safe(prov)
    fixed = raw.replace("\\n", "\n")  # 把字面 \n 轉成換行
    # 強制成正確內容
    fixed = 'from .router import fetch_bars\n\n__all__ = ["fetch_bars"]\n'
    write_text_utf8(prov, fixed)

# 3) test/typing、ws_data_received、Iterable
tests_handlers = ROOT / "tests" / "test_handlers.py"
if tests_handlers.exists():
    ensure_import(tests_handlers, "from typing import Any, Dict")

api_test = ROOT / "scripts" / "test_api_connection.py"
if api_test.exists():
    txt = read_text_safe(api_test)
    if "ws_data_received" not in txt:
        txt = "ws_data_received = False\n" + txt
        write_text_utf8(api_test, txt)

binance = ROOT / "src" / "quantproject" / "data_pipeline" / "backends" / "binance.py"
if binance.exists():
    ensure_import(binance, "from typing import Iterable")

# 4) 幫兩個 global 收尾
main_trading = ROOT / "main_trading.py"
if main_trading.exists():
    regex_sub_file(
        main_trading,
        r"^\s*global\s+trading_system.*$",
        "# global trading_system  # removed by ci-hotfix",
    )
    ensure_header_def(main_trading, "trading_system", "None")

mt4_cfg = ROOT / "mt4_bridge" / "config.py"
if mt4_cfg.exists():
    regex_sub_file(
        mt4_cfg,
        r"^\s*global\s+_global_config.*$",
        "# global _global_config  # removed by ci-hotfix",
    )
    ensure_header_def(mt4_cfg, "_global_config", "None")

# 5) 把 model_trainer.py 轉成 UTF-8
model_trainer = ROOT / "src" / "models" / "ml_models" / "model_trainer.py"
if model_trainer.exists():
    text = read_text_safe(model_trainer)
    write_text_utf8(model_trainer, text)

# 6) 刪除 tmp_fix2.py（避免 parser error）
tmp_fix2 = ROOT / "tmp_fix2.py"
if tmp_fix2.exists():
    tmp_fix2.unlink()

print("[hotfix] done")
