@echo off
title Ultra-Fast Alpaca Training
echo ===============================================
echo    ULTRA-FAST ALPACA TRAINING (Under 10 mins)
echo ===============================================
echo.
echo Starting ultra-fast training...
echo.

cd /d "%~dp0"
call alpaca_llm_env\Scripts\activate.bat
python scripts\ultra_fast_train.py

echo.
echo Training completed! Check the results above.
pause
