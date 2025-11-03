@echo off
echo ============================================================
echo         TRADING SYSTEM MONITOR
echo ============================================================
echo.

REM Set Python encoding to UTF-8
set PYTHONIOENCODING=utf-8

REM Change to project directory
cd /d C:\Users\niuji\Documents\QuantProject

echo Monitoring trading system status...
echo Press Ctrl+C to stop monitoring
echo.

python monitor_live_trading.py

pause