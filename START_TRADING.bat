@echo off
echo ============================================================
echo         AUTOMATED TRADING SYSTEM LAUNCHER
echo ============================================================
echo.
echo Starting Quantitative Trading System...
echo.
echo Configuration:
echo - Monitoring: 4,215 stocks
echo - Max Positions: 20
echo - Risk per trade: 5%%
echo - Stop Loss: 5%%
echo - Take Profit: 10%%
echo.
echo ============================================================
echo.

REM Set Python encoding to UTF-8
set PYTHONIOENCODING=utf-8

REM Change to project directory
cd /d C:\Users\niuji\Documents\QuantProject

REM Create necessary directories
if not exist logs mkdir logs
if not exist data mkdir data

echo [1] Starting trading system...
echo.

REM Start the trading system
python live_trading_system_full.py

echo.
echo ============================================================
echo Trading system stopped.
echo ============================================================
pause