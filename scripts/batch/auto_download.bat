@echo off
echo ============================================================
echo STARTING AUTOMATIC DATA DOWNLOAD
echo ============================================================
echo.
echo This window will download all 4,215 stocks automatically
echo DO NOT CLOSE THIS WINDOW
echo.

cd /d C:\Users\niuji\Documents\QuantProject

python -c "import os; os.system('cls'); exec(open('auto_download_worker.py').read())"

pause