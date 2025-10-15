"""
Fix requirements.txt to use pinned versions
"""

import subprocess
import sys
from pathlib import Path


def get_installed_version(package_name):
    """Get installed version of a package"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
    except:
        pass
    return None


def fix_requirements():
    """Fix requirements.txt with pinned versions"""

    # Essential packages with specific versions for stability
    requirements = """# requirements.txt
# Python 3.9+
# Fixed versions for stability and security

# Core Dependencies
numpy==1.24.3
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.0

# Data Providers
yfinance==0.2.28
alpha-vantage==2.3.1
requests==2.31.0
python-dotenv==1.0.0

# Data Processing
pandas-ta==0.3.14b0
statsmodels==0.14.0

# Visualization
plotly==5.17.0
matplotlib==3.7.2
seaborn==0.12.2

# API & Web
aiohttp==3.8.5
websockets==11.0.3
httpx==0.24.1
websocket-client==1.6.1

# Utilities
pyyaml==6.0.1
click==8.1.7
colorama==0.4.6
tqdm==4.66.1

# Database
sqlalchemy==2.0.21
sqlite3-api==2.0.0

# Testing
pytest==7.4.2
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Logging
loguru==0.7.2

# Optional - Machine Learning (comment out if not needed)
# torch==2.0.1
# tensorflow==2.13.0
# stable-baselines3==2.1.0
# gym==0.26.2
# optuna==3.3.0

# Optional - Advanced Analytics (comment out if not needed)  
# ta-lib==0.4.28  # Requires separate C library installation
# networkx==3.1
# torch-geometric==2.3.1
"""

    # Write the fixed requirements
    req_path = Path(__file__).parent.parent / "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(requirements)

    print(f"[OK] Fixed requirements.txt with pinned versions")

    # Create requirements-dev.txt for development dependencies
    requirements_dev = """# requirements-dev.txt
# Development dependencies

# Code Quality
black==23.9.1
flake8==6.1.0
mypy==1.5.1
pylint==2.17.5
isort==5.12.0

# Testing
pytest==7.4.2
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0

# Jupyter/Notebooks
jupyter==1.0.0
ipykernel==6.25.2
nbconvert==7.8.0

# Debugging
ipdb==0.13.13
"""

    req_dev_path = Path(__file__).parent.parent / "requirements-dev.txt"
    with open(req_dev_path, "w", encoding="utf-8") as f:
        f.write(requirements_dev)

    print(f"[OK] Created requirements-dev.txt")

    # Create requirements-minimal.txt for minimal installation
    requirements_minimal = """# requirements-minimal.txt
# Minimal requirements for basic functionality

numpy==1.24.3
pandas==2.0.3
yfinance==0.2.28
requests==2.31.0
python-dotenv==1.0.0
plotly==5.17.0
loguru==0.7.2
"""

    req_minimal_path = Path(__file__).parent.parent / "requirements-minimal.txt"
    with open(req_minimal_path, "w", encoding="utf-8") as f:
        f.write(requirements_minimal)

    print(f"[OK] Created requirements-minimal.txt")

    return True


def main():
    print("\n" + "=" * 60)
    print("FIXING REQUIREMENTS.TXT")
    print("=" * 60)

    success = fix_requirements()

    if success:
        print("\n[SUCCESS] Requirements fixed!")
        print("\nNext steps:")
        print("1. Install minimal requirements:")
        print("   pip install -r requirements-minimal.txt")
        print("\n2. For full installation:")
        print("   pip install -r requirements.txt")
        print("\n3. For development:")
        print("   pip install -r requirements-dev.txt")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
