@echo off
echo ========================================================
echo     QUANTTRADING ANALYTICS DASHBOARD LAUNCHER
echo ========================================================
echo.
echo Starting Analytics Dashboard...
echo Dashboard will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo ========================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Launch the analytics dashboard
python launch_analytics_dashboard.py

echo.
echo Dashboard stopped.
pause