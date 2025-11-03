import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 保底解碼
        return p.read_bytes().decode("utf-8", errors="replace")


def write(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")


# 1) config/config.py: 修掉 "✓" + API Key loaded ...
cfg = ROOT / "config" / "config.py"
if cfg.exists():
    s = read(cfg)
    # 極端字串樣式一律矯正
    s = s.replace(
        'logger.info("" +  API Key loaded: {CAPITAL_API_KEY[:4]}...{CAPITAL_API_KEY[-4:]}")',
        'logger.info(f" API Key loaded: {CAPITAL_API_KEY[:4]}...{CAPITAL_API_KEY[-4:]}")',
    )
    # 兜底：任何包含 API Key loaded 的 logger.info 直接標準化
    s = re.sub(
        r"^\s*logger\.info\([^\n]*API\s*Key[^\n]*loaded[^\n]*\)\s*$",
        '    logger.info(f" API Key loaded: {CAPITAL_API_KEY[:4]}...{CAPITAL_API_KEY[-4:]}")',
        s,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    write(cfg, s)

# 2) src/connectors/capital_com_api.py: 修 "" + Authentication successful!
cap = ROOT / "src" / "connectors" / "capital_com_api.py"
if cap.exists():
    s = read(cap)
    s = s.replace(
        'logger.info("" +  Authentication successful!")',
        'logger.info(" Authentication successful!")',
    )
    # 再保險一次
    s = re.sub(
        r'logger\.info\(\s*""\s*\+\s*Authentication\s*successful!\s*\)',
        'logger.info(" Authentication successful!")',
        s,
        flags=re.IGNORECASE,
    )
    # Identifier 一併標準化
    s = re.sub(
        r'logger\.info\(\s*Identifier:\s*\{api\.identifier\}"\s*\)',
        'logger.info("Identifier: %s", api.identifier)',
        s,
    )
    write(cap, s)

# 3) tests/test_portfolio_env.py: 補 obs 與 symbols
tpe = ROOT / "tests" / "test_portfolio_env.py"
if tpe.exists():
    s = read(tpe)
    # 單獨一行的清單 -> 宣告成 symbols
    s = re.sub(r"^(\s*)\[(.*?)\]\s*$", r"\1symbols = [\2]", s, flags=re.MULTILINE)
    # 沒有指派就呼叫 observation：改成賦值
    s = re.sub(
        r"^(\s*)self\.env\._get_observation\(\)\s*$",
        r"\1obs = self.env._get_observation()",
        s,
        flags=re.MULTILINE,
    )
    # 在使用 predict(obs, ...) 前最近 10 行沒有 obs 賦值就插入一行
    lines = s.splitlines()
    i = 0
    while i < len(lines):
        if "predict(obs" in lines[i]:
            window = "\n".join(lines[max(0, i - 10) : i])
            if "obs =" not in window:
                indent = re.match(r"^(\s*)", lines[i]).group(1)
                lines.insert(i, f"{indent}obs = self.env._get_observation()")
                i += 1
        i += 1
    s = "\n".join(lines)
    write(tpe, s)

# 4) tools/patch_turnover_yaml.py: 補 data 變數
pty = ROOT / "tools" / "patch_turnover_yaml.py"
if pty.exists():
    s = read(pty)
    s = re.sub(
        r"yaml\.safe_load\((.*?)\)\s*or\s*\{\}", r"data = yaml.safe_load(\1) or {}", s
    )
    write(pty, s)

# 5) ultra_simple_ppo.py: 定義 symbols
usp = ROOT / "ultra_simple_ppo.py"
if usp.exists():
    s = read(usp)
    s = re.sub(r'^(\s*)\[(.*"QQQ".*)\]\s*$', r"\1symbols = [\2]", s, flags=re.MULTILINE)
    # 如果仍找不到 symbols，就給個預設
    if "symbols =" not in s:
        s = s.replace(
            "# Download a few stocks that definitely work",
            '# Download a few stocks that definitely work\nsymbols = ["AAPL","MSFT","GOOGL","AMZN","TSLA","SPY","QQQ"]',
        )
    write(usp, s)

# 6) validate_stock_symbols.py: 修 data 未定義
vss = ROOT / "validate_stock_symbols.py"
if vss.exists():
    s = read(vss)
    # json.load -> 指派到 data
    s = re.sub(
        r"with open\([^\n]*\)\s+as f:\s*\n\s*json\.load\(f\)",
        lambda m: m.group(0).replace("json.load(f)", "data = json.load(f)"),
        s,
    )
    # 安全檢查區塊：避免 data 未定義
    s = s.replace(
        "if not data.empty and len(data) > 5:",
        "cond_ok = False\n        "
        'if "data" in locals():\n            '
        'cond_ok = (hasattr(data, "empty") and (not data.empty) and len(data) > 5) or (isinstance(data, (list, tuple)) and len(data) > 5)\n        '
        "if cond_ok:",
    )
    s = s.replace("return True, len(data)", "return True, len(data)")
    write(vss, s)

print("[hardfix] undefined names & logger strings patched")
