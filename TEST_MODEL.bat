@echo off
echo ========================================
echo    ALPACA DOMINATION TESTING SUITE
echo ========================================
echo.
echo Choose your testing method:
echo.
echo 1. 🌐 Web Interface (Browser Chat)
echo 2. 🧪 Terminal Testing (Command Line)
echo 3. 📊 Both Web + Terminal
echo 4. ⚙️ Check Model Status
echo 5. 🔄 Restart Training
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🌐 Starting Web Interface...
    echo Opening browser at http://localhost:8000/chat
    start http://localhost:8000/chat
    echo.
    echo Starting API server...
    cd backend
    ..\venv\Scripts\python.exe main.py
) else if "%choice%"=="2" (
    echo.
    echo 🧪 Starting Terminal Tester...
    .\venv\Scripts\python.exe terminal_tester.py
) else if "%choice%"=="3" (
    echo.
    echo 📊 Starting both Web and Terminal...
    echo Opening web interface...
    start http://localhost:8000/chat
    echo.
    echo Starting API server in background...
    start "API Server" cmd /c "cd backend && ..\venv\Scripts\python.exe main.py"
    echo.
    echo Waiting for server to start...
    timeout /t 3 /nobreak > nul
    echo.
    echo Starting terminal tester...
    .\venv\Scripts\python.exe terminal_tester.py
) else if "%choice%"=="4" (
    echo.
    echo ⚙️ Checking model status...
    if exist "ultra_fast_model\model.safetensors" (
        echo ✅ Model found: ultra_fast_model/
        dir ultra_fast_model\model.safetensors
        echo.
        echo 🧪 Quick model test...
        .\venv\Scripts\python.exe -c "from pathlib import Path; print('Model size:', Path('ultra_fast_model/model.safetensors').stat().st_size // 1024 // 1024, 'MB')"
    ) else (
        echo ❌ No trained model found!
        echo Please run training first.
    )
    pause
) else if "%choice%"=="5" (
    echo.
    echo 🔄 Launching training menu...
    .\RUN_PROJECT.bat
) else (
    echo ❌ Invalid choice. Please try again.
    pause
)
