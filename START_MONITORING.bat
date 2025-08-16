@echo off
echo.
echo ================================================================================
echo                   4000+ STOCK MONITORING SYSTEM LAUNCHER
echo ================================================================================
echo.

REM Set UTF-8 encoding for Python output
set PYTHONIOENCODING=utf-8

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.7+ and add it to PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import pandas, numpy, yfinance, yaml" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo Installing required packages...
    pip install pandas numpy yfinance pyyaml python-dotenv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
)

REM Generate stock list if needed
if not exist "data\all_symbols.txt" (
    echo.
    echo Generating stock list...
    python get_all_us_stocks.py
)

REM Start the monitoring system
echo.
echo Starting 4000+ Stock Monitoring System...
echo Press Ctrl+C to stop
echo.
echo ================================================================================
python START_4000_STOCKS_FINAL.py

echo.
echo Monitoring system stopped.
pause