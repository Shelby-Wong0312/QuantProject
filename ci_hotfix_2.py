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
    f = ROOT / "config" / "config.py"
    if not f.exists():
        return
    s = read(f)
    # 這幾種都給我變成人話
    s = s.replace(
        'logger.warning(".env file not found in project directory!")',
        'logger.warning(".env file not found in project directory!")',
    )
    s = s.replace(
        'logger.error("API credentials not found!")', 'logger.error("API credentials not found!")'
    )
    # 偶爾有人愛加 "" + 純字串，幫你收斂
    s = re.sub(
        r'logger\.info\(\s*""\s*\+\s*Found\s+\.env\s+file\s+at:\s*\{env_file_path\}"\s*\)',
        'logger.info(" Found .env file at: %s", env_file_path)',
        s,
    )
    write(f, s)


def fix_capital_api():
    f = ROOT / "src" / "connectors" / "capital_com_api.py"
    if not f.exists():
        return
    s = read(f)
    # 設法把「"" + Authentication successful!」變正常
    s = re.sub(
        r'logger\.info\(\s*""\s*\+\s*Authentication successful!"\s*\)',
        'logger.info(" Authentication successful!")',
        s,
    )
    # 之前那條 Identifier 也順手穩一點
    s = re.sub(
        r'logger\.info\(\s*Identifier:\s*\{api\.identifier\}"\s*\)',
        'logger.info("Identifier: %s", api.identifier)',
        s,
    )
    write(f, s)


def main():
    fix_config()
    fix_capital_api()
    print("[hotfix-2] fixed logger strings")


if __name__ == "__main__":
    main()
