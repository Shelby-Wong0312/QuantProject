@echo off
cls
echo.
echo ================================================================================
echo                    PPO FULL TRAINING - 4000 STOCKS
echo                    Using Maximum Historical Data (2010-2025)
echo ================================================================================
echo.
echo WARNING: This will take several hours to complete!
echo.
echo Training Configuration:
echo - Stocks: 4000 symbols
echo - Date Range: 2010-01-01 to Today (15 years)
echo - Iterations: 1000
echo - Feature Dimensions: 220
echo - Action Space: 4 (Hold, Buy, Sell, Strong Signal)
echo.
echo Press Ctrl+C anytime to stop and save progress.
echo.
pause
echo.
echo Starting training...
python TRAIN_PPO_FULL.py
echo.
echo Training complete! Check ppo_full_final.pt for the trained model.
pause