@echo off
echo ========================================
echo Starting Live Trading System
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip install -q yfinance pandas numpy plotly python-dotenv 2>nul

echo [2/3] Loading environment variables...
if not exist .env (
    echo WARNING: .env file not found!
    echo Please configure your API keys in .env file
    pause
)

echo [3/3] Starting live trading system...
echo.
echo ========================================
echo LIVE TRADING ACTIVE
echo ========================================
echo.
echo Press Ctrl+C to stop trading
echo.

python -c "import asyncio; from src.live_trading.live_system import main; asyncio.run(main())"

echo.
echo ========================================
echo Trading system stopped
echo ========================================
pause