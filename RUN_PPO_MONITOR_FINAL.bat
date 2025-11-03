@echo off
title PPO Unified Monitor
color 0A
mode con: cols=105 lines=50
cls

echo ====================================================================================================
echo                                  PPO UNIFIED MONITORING SYSTEM
echo ====================================================================================================
echo.
echo Initializing PPO model and starting real-time stock monitoring...
echo This system will monitor all configured stocks and provide trading signals.
echo.
echo Press Ctrl+C to stop monitoring
echo.
echo ----------------------------------------------------------------------------------------------------

python PPO_MONITOR_FINAL.py

echo.
echo Monitoring stopped.
pause