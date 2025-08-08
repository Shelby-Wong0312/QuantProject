@echo off
echo Starting historical data download...
echo This will download 15 years of data for 4,215 stocks
echo The process will run in background
echo.
echo y | python download_long_term_data.py > download.log 2>&1
echo Download complete. Check download.log for details.