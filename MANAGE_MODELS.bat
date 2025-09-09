@echo off
echo ========================================
echo        SAVED MODELS MANAGER
echo ========================================
echo.
echo Your trained models:
echo.

if exist "ultra_fast_model" (
    echo ✅ ULTRA-FAST MODEL - ultra_fast_model/
    dir ultra_fast_model\*.safetensors /b 2>nul | findstr . >nul && echo    └─ Model file found
    echo    └─ Size: ~497MB
    echo.
)

if exist "lightning_model" (
    echo ✅ LIGHTNING MODEL - lightning_model/
    dir lightning_model\*.safetensors /b 2>nul | findstr . >nul && echo    └─ Model file found
    echo.
)

if exist "trained_models" (
    echo ✅ FULL MODEL - trained_models/
    dir trained_models\*.safetensors /s /b 2>nul | findstr . >nul && echo    └─ Model file found
    echo.
)

echo ========================================
echo        ACTIONS AVAILABLE
echo ========================================
echo.
echo 1. Test Ultra-Fast Model
echo 2. Load Model in API Server
echo 3. View Model Details
echo 4. Train New Model
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🧪 Testing your saved model...
    .\venv\Scripts\python.exe test_saved_model.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting API server with your model...
    .\venv\Scripts\python.exe backend\main.py
) else if "%choice%"=="3" (
    echo.
    echo 📊 Model details:
    .\venv\Scripts\python.exe -c "from pathlib import Path; import json; p=Path('ultra_fast_model/config.json'); print(json.load(open(p)) if p.exists() else 'No model config found')"
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🎯 Starting training launcher...
    .\RUN_PROJECT.bat
) else (
    echo Invalid choice.
    pause
)
