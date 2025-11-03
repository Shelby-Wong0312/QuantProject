@echo off
echo Starting batch validation of all stocks...
echo This will take several hours to complete.
echo Progress will be saved to batch_validation_checkpoint.json
echo.
echo Press Ctrl+C to stop (progress will be saved)
echo.

:loop
python batch_validate.py
if %errorlevel% neq 0 (
    echo.
    echo Validation stopped or encountered an error.
    echo Waiting 10 seconds before retry...
    timeout /t 10 /nobreak
    goto loop
)

echo.
echo Validation completed successfully!
echo Check FINAL_VALIDATION_REPORT.json for results.
pause