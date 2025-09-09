@echo off
echo ========================================
echo    COMPLETE LLM PROJECT LAUNCHER
echo ========================================
echo.
echo Choose your training option:
echo.
echo 1. ULTRA-FAST Training (2-3 minutes)
echo 2. LIGHTNING Training (5-7 minutes) 
echo 3. FULL Training (10+ minutes)
echo 4. Skip Training - Launch Web Interface
echo 5. Launch API Server Only
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting ULTRA-FAST training...
    .\venv\Scripts\python.exe scripts\speed_train.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo ⚡ Starting LIGHTNING training...
    .\venv\Scripts\python.exe scripts\lightning_train.py
    pause
) else if "%choice%"=="3" (
    echo.
    echo 🎯 Starting FULL training...
    .\venv\Scripts\python.exe scripts\train_model.py
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🌐 Launching Web Interface...
    start http://localhost:8000
    echo Web interface will be available at http://localhost:8000
    pause
) else if "%choice%"=="5" (
    echo.
    echo 🔌 Starting API Server...
    .\venv\Scripts\python.exe backend\main.py
) else (
    echo Invalid choice. Please run again.
    pause
)
