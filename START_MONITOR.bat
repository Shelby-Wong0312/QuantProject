@echo off
title PPO Stock Monitor
color 0A
cls

echo ================================================================================
echo                           PPO STOCK MONITORING SYSTEM
echo ================================================================================
echo.
echo Loading PPO model and starting real-time monitoring...
echo.

python PPO_LIVE_MONITOR.py

pause