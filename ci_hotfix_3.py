import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        for enc in ("utf-8-sig", "cp950", "big5", "latin1"):
            try:
                return p.read_text(encoding=enc)
            except Exception:
                pass
        return p.read_bytes().decode("utf-8", errors="replace")


def write(p: Path, s: str):
    p.write_text(s, encoding="utf-8", newline="\n")


def fix_config():
    p = ROOT / "config" / "config.py"
    if not p.exists():
        return
    s = read(p)
    before = s
    # "" + API Key loaded: {...}...{...}  -> f-string 一句話
    s = re.sub(
        r'logger\.info\(\s*""\s*\+\s*API\s*Key\s*loaded:[^\)]*\)',
        'logger.info(f"✓ API Key loaded: {CAPITAL_API_KEY[:4]}...{CAPITAL_API_KEY[-4:]}")',
        s,
        flags=re.IGNORECASE,
    )
    # Found .env 也順便保險處理
    s = re.sub(
        r'logger\.info\(\s*""\s*\+\s*Found[^\)]*\)',
        'logger.info(" Found .env file at: %s", env_file_path)',
        s,
        flags=re.IGNORECASE,
    )
    # 其他警告/錯誤訊息歸位
    s = re.sub(
        r"logger\.warning\([^)]*\.env[^)]*not[^)]*found[^)]*\)",
        'logger.warning(".env file not found in project directory!")',
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"logger\.error\([^)]*API[^)]*credentials[^)]*not[^)]*found[^)]*\)",
        'logger.error("API credentials not found!")',
        s,
        flags=re.IGNORECASE,
    )
    if s != before:
        write(p, s)


def fix_capital_api():
    p = ROOT / "src" / "connectors" / "capital_com_api.py"
    if not p.exists():
        return
    s = read(p)
    before = s
    # "" + Authentication successful!
    s = re.sub(
        r'logger\.info\(\s*""\s*\+\s*Authentication\s*successful!\s*\)',
        'logger.info(" Authentication successful!")',
        s,
        flags=re.IGNORECASE,
    )
    # Identifier 也確保是安全格式
    s = re.sub(
        r'logger\.info\(\s*Identifier:\s*\{api\.identifier\}"\s*\)',
        'logger.info("Identifier: %s", api.identifier)',
        s,
    )
    if s != before:
        write(p, s)


if __name__ == "__main__":
    fix_config()
    fix_capital_api()
    print("[hotfix-3] normalized remaining logger strings")
