@echo off
echo ============================================================
echo CHECKING DATA DOWNLOAD STATUS
echo ============================================================
echo.

REM Check database record count
python -c "import sqlite3; conn = sqlite3.connect('data/quant_trading.db'); c = conn.cursor(); c.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM daily_data'); r = c.fetchone(); print('Current Status: %d stocks, %d records' % (r[0], r[1])); conn.close()"

echo.
echo Waiting 10 seconds...
timeout /t 10 /nobreak > nul

REM Check again
python -c "import sqlite3; conn = sqlite3.connect('data/quant_trading.db'); c = conn.cursor(); c.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM daily_data'); r = c.fetchone(); print('After 10 sec: %d stocks, %d records' % (r[0], r[1])); conn.close()"

echo.
echo Check the new "Data Download" window for live progress!
echo.
pause