@echo off
echo ===================================================
echo     VoiceGuard AI Detection Server Launcher
echo ===================================================

echo [1/2] Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements. Please check Python installation.
    pause
    exit /b
)

echo.
echo [2/2] Starting VoiceGuard Server...
echo Server will be available at http://127.0.0.1:8000
echo.
python main.py

pause
