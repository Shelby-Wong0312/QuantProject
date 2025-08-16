"""
Stage 0: Security Fix Script
Fix all hardcoded credentials and security issues
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def find_hardcoded_credentials(file_path: Path) -> List[Tuple[int, str]]:
    """Find hardcoded credentials in a file"""
    issues = []
    
    # Patterns to search for hardcoded credentials
    patterns = [
        (r"os\.environ\['CAPITAL_API_KEY'\]\s*=\s*'[^']+'", "Hardcoded API Key"),
        (r"os\.environ\['CAPITAL_IDENTIFIER'\]\s*=\s*'[^']+'", "Hardcoded Identifier"),
        (r"os\.environ\['CAPITAL_API_PASSWORD'\]\s*=\s*'[^']+'", "Hardcoded Password"),
        (r"'kugBoHCUcjaaNwGV'", "Hardcoded API Key Value"),
        (r"'niujinheitaizi@gmail\.com'", "Hardcoded Email"),
        (r"'@Nickatnyte3'", "Hardcoded Password Value"),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines, 1):
            for pattern, desc in patterns:
                if re.search(pattern, line):
                    issues.append((i, f"Line {i}: {desc} - {line.strip()[:80]}"))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return issues

def fix_hardcoded_credentials(file_path: Path) -> bool:
    """Fix hardcoded credentials in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Replace hardcoded credentials with environment variable reads
        replacements = [
            # Remove hardcoded assignments
            (r"os\.environ\['CAPITAL_API_KEY'\]\s*=\s*'kugBoHCUcjaaNwGV'", "# os.environ['CAPITAL_API_KEY'] removed - use .env file"),
            (r"os\.environ\['CAPITAL_IDENTIFIER'\]\s*=\s*'niujinheitaizi@gmail\.com'", "# os.environ['CAPITAL_IDENTIFIER'] removed - use .env file"),
            (r"os\.environ\['CAPITAL_API_PASSWORD'\]\s*=\s*'@Nickatnyte3'", "# os.environ['CAPITAL_API_PASSWORD'] removed - use .env file"),
            
            # For direct usage, replace with os.getenv()
            (r"'kugBoHCUcjaaNwGV'", "os.getenv('CAPITAL_API_KEY')"),
            (r"'niujinheitaizi@gmail\.com'", "os.getenv('CAPITAL_IDENTIFIER')"),
            (r"'@Nickatnyte3'", "os.getenv('CAPITAL_API_PASSWORD')"),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def create_env_example():
    """Create .env.example file"""
    env_example = """# Capital.com API Configuration
CAPITAL_API_KEY=your_api_key_here
CAPITAL_IDENTIFIER=your_email_here
CAPITAL_API_PASSWORD=your_password_here
CAPITAL_DEMO_MODE=True

# Alpaca Markets Configuration (Free)
ALPACA_API_KEY_ID=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_END_POINT=https://paper-api.alpaca.markets/v2

# Alpha Vantage Configuration (Free)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Yahoo Finance (No API key needed)
# Uses yfinance library - unlimited free access

# System Configuration
LOG_LEVEL=INFO
MAX_POSITIONS=50
RISK_LIMIT=0.02
"""
    
    env_example_path = project_root / '.env.example'
    with open(env_example_path, 'w', encoding='utf-8') as f:
        f.write(env_example)
    print(f"[OK] Created {env_example_path}")

def update_gitignore():
    """Update .gitignore to exclude sensitive files"""
    gitignore_path = project_root / '.gitignore'
    
    # Lines to add if not present
    sensitive_patterns = [
        '.env',
        '.env.local',
        '*.pem',
        '*.key',
        '*.crt',
        'api_keys.json',
        'credentials.json',
        '**/secrets/*',
        'config/api_config_live.json',
    ]
    
    existing = set()
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f.readlines())
    
    with open(gitignore_path, 'a', encoding='utf-8') as f:
        for pattern in sensitive_patterns:
            if pattern not in existing:
                f.write(f"\n{pattern}")
    
    print(f"[OK] Updated .gitignore")

def main():
    print("\n" + "="*80)
    print("STAGE 0: SECURITY FIX")
    print("="*80)
    
    # Files with known hardcoded credentials
    files_to_fix = [
        'close_oil_positions.py',
        'live_trading_system_full.py',
        'buy_wti_oil.py',
        'execute_wti_trade.py',
        'search_oil_markets.py',
        'sell_wti_oil.py',
    ]
    
    print("\n[1] Scanning for hardcoded credentials...")
    total_issues = 0
    
    for file_name in files_to_fix:
        file_path = project_root / file_name
        if file_path.exists():
            issues = find_hardcoded_credentials(file_path)
            if issues:
                print(f"\n  {file_name}:")
                for _, desc in issues:
                    print(f"    - {desc}")
                total_issues += len(issues)
    
    if total_issues > 0:
        print(f"\n[!] Found {total_issues} security issues")
        
        print("\n[2] Fixing hardcoded credentials...")
        fixed_count = 0
        
        for file_name in files_to_fix:
            file_path = project_root / file_name
            if file_path.exists():
                if fix_hardcoded_credentials(file_path):
                    print(f"  [FIXED] {file_name}")
                    fixed_count += 1
        
        print(f"\n[OK] Fixed {fixed_count} files")
    else:
        print("\n[OK] No hardcoded credentials found")
    
    print("\n[3] Creating .env.example...")
    create_env_example()
    
    print("\n[4] Updating .gitignore...")
    update_gitignore()
    
    # Check if .env exists
    env_path = project_root / '.env'
    if not env_path.exists():
        print("\n[!] WARNING: .env file not found")
        print("    Please copy .env.example to .env and add your credentials")
    else:
        print("\n[OK] .env file exists")
    
    print("\n[5] Security Recommendations:")
    print("  1. Never commit .env file to git")
    print("  2. Use environment variables for all credentials")
    print("  3. Rotate API keys regularly")
    print("  4. Use different keys for dev/staging/production")
    print("  5. Enable IP allowlisting where possible")
    
    print("\n" + "="*60)
    print("SECURITY FIX COMPLETE")
    print("="*60)
    
    # Verify fixes
    print("\n[6] Verifying fixes...")
    remaining_issues = 0
    for file_name in files_to_fix:
        file_path = project_root / file_name
        if file_path.exists():
            issues = find_hardcoded_credentials(file_path)
            if issues:
                remaining_issues += len(issues)
    
    if remaining_issues == 0:
        print("[SUCCESS] All hardcoded credentials removed!")
    else:
        print(f"[WARNING] {remaining_issues} issues remain")
    
    return remaining_issues == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)