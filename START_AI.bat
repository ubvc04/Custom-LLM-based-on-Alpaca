@echo off
echo ===============================================
echo        🚀 YOUR AI PROJECT LAUNCHER 🚀
echo ===============================================
echo.
echo Your AI Status: 
echo   ✅ Model: Trained (497MB Alpaca model)
echo   ✅ Server: Running (if started)
echo   ✅ Ready: For chat and testing
echo.
echo ===============================================
echo           CHOOSE HOW TO RUN:
echo ===============================================
echo.
echo 1. 🌐 START AI CHAT (Web Interface)
echo 2. 🧪 TERMINAL TESTING (Command Line)
echo 3. 🔄 RESTART AI SERVER
echo 4. 📊 CHECK AI STATUS
echo 5. 🎯 TRAIN NEW MODEL
echo 6. 📖 VIEW ALL ENDPOINTS
echo 7. ❌ EXIT
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo 🌐 Starting AI Web Chat...
    echo Opening: http://localhost:8000/chat
    start http://localhost:8000/chat
    echo.
    echo If not already running, starting AI server...
    .\venv\Scripts\python.exe simple_ai_server.py
    
) else if "%choice%"=="2" (
    echo.
    echo 🧪 Starting Terminal Testing...
    .\venv\Scripts\python.exe terminal_tester.py
    
) else if "%choice%"=="3" (
    echo.
    echo 🔄 Restarting AI Server...
    taskkill /F /IM python.exe /T 2>nul
    echo Starting fresh AI server...
    .\venv\Scripts\python.exe simple_ai_server.py
    
) else if "%choice%"=="4" (
    echo.
    echo 📊 Checking AI Status...
    echo.
    echo 🤖 AI Model Files:
    if exist "ultra_fast_model\model.safetensors" (
        echo ✅ Model found: ultra_fast_model/
        for %%F in (ultra_fast_model\model.safetensors) do echo    Size: %%~zF bytes
    ) else (
        echo ❌ No model found! Please train first.
    )
    echo.
    echo 🔍 Running Processes:
    tasklist | findstr python.exe
    echo.
    echo 🌐 Testing Connection:
    .\venv\Scripts\python.exe -c "import requests; r=requests.get('http://localhost:8000/', timeout=2); print('✅ AI Server: ONLINE' if r.status_code==200 else '❌ AI Server: OFFLINE')" 2>nul || echo ❌ AI Server: OFFLINE
    pause
    
) else if "%choice%"=="5" (
    echo.
    echo 🎯 Opening Training Menu...
    .\RUN_PROJECT.bat
    
) else if "%choice%"=="6" (
    echo.
    echo 📖 YOUR AI ENDPOINTS:
    echo ===============================================
    echo 🌐 Web Chat:     http://localhost:8000/chat
    echo 🔌 API:          http://localhost:8000/v1/chat/completions
    echo 📊 Status:       http://localhost:8000/
    echo 📚 Docs:         http://localhost:8000/docs
    echo ===============================================
    echo.
    echo 💡 USAGE EXAMPLES:
    echo.
    echo Web Chat: Just open the URL and type messages
    echo API Call:  curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
    echo.
    pause
    
) else if "%choice%"=="7" (
    echo.
    echo 👋 Goodbye!
    exit
    
) else (
    echo.
    echo ❌ Invalid choice. Please try again.
    pause
    goto :EOF
)

echo.
echo 🔄 Returning to menu...
pause
goto :EOF
