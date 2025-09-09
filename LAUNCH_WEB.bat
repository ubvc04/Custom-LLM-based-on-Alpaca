@echo off
echo ========================================
echo    LAUNCHING WEB INTERFACE + API
echo ========================================
echo.
echo Starting API Server...
start "API Server" .\venv\Scripts\python.exe backend\main.py
echo.
echo Waiting 5 seconds for API to start...
timeout /t 5 /nobreak > nul
echo.
echo Opening web interface...
start http://localhost:8000
echo.
echo ✅ Project is now running!
echo.
echo 🔌 API Server: http://localhost:8000
echo 🌐 Web Interface: http://localhost:8000
echo 📖 API Docs: http://localhost:8000/docs
echo.
pause
