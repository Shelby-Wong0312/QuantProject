@echo off
echo ============================================================
echo      LAUNCHING COMPLETE TRADING SYSTEM
echo ============================================================
echo.

REM Set encoding
set PYTHONIOENCODING=utf-8

REM Change to project directory  
cd /d C:\Users\niuji\Documents\QuantProject

echo [1] Starting Trading System (4,215 stocks)...
start "TRADING SYSTEM" cmd /k python simple_trading_system.py

timeout /t 3 /nobreak > nul

echo [2] Starting Live Dashboard...
start "LIVE DASHBOARD" cmd /k python live_dashboard.py

echo.
echo ============================================================
echo SYSTEM LAUNCHED SUCCESSFULLY!
echo ============================================================
echo.
echo Two windows should be open:
echo 1. TRADING SYSTEM - Executing trades
echo 2. LIVE DASHBOARD - Monitoring status
echo.
echo If you don't see the windows, check your taskbar.
echo.
pause