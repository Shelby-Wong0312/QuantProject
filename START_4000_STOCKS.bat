@echo off
echo ================================================================================
echo                     4000+ 股票大規模監控交易系統
echo ================================================================================
echo.

REM 檢查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到Python！
    echo 請安裝Python 3.7+
    pause
    exit /b 1
)

echo [1/3] 檢查依賴套件...
pip install -q yfinance pandas numpy pyzmq python-dotenv 2>nul
if errorlevel 1 (
    echo [警告] 某些套件可能未正確安裝
)

echo [2/3] 檢查環境設定...
if not exist .env (
    echo [警告] 找不到 .env 文件！
    echo 請配置您的API金鑰
)

echo.
echo ================================================================================
echo 請選擇操作：
echo.
echo   1. 啟動4000+股票監控系統
echo   2. 創建4000股票列表文件
echo   3. 測試系統（50檔股票）
echo   4. 退出
echo.
set /p choice="請輸入選項 (1-4): "

if "%choice%"=="1" goto start_full
if "%choice%"=="2" goto create_list
if "%choice%"=="3" goto test_mode
if "%choice%"=="4" goto exit

:start_full
echo.
echo ================================================================================
echo                        啟動完整監控系統
echo ================================================================================
echo.
echo 正在啟動4000+股票監控...
echo 按 Ctrl+C 停止系統
echo.
python start_4000_stocks_monitoring.py
goto end

:create_list
echo.
echo 創建股票列表...
python start_4000_stocks_monitoring.py --create-list
echo.
echo 股票列表已創建！
pause
goto end

:test_mode
echo.
echo ================================================================================
echo                         測試模式（50檔股票）
echo ================================================================================
echo.
python start_live_trading_simple.py
goto end

:exit
echo 退出系統
goto end

:end
echo.
echo ================================================================================
echo 系統已停止
echo ================================================================================
pause