@echo off
echo LIGHTNING-FAST Alpaca Training (Under 5 minutes!)
echo ================================================

cd /d "c:\Users\baves\Downloads\Custom LLM"

echo Activating environment...
call alpaca_llm_env\Scripts\activate.bat

echo Starting lightning training...
python scripts\lightning_train.py

echo Training complete!
pause
