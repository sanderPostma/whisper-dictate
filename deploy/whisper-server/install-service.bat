@echo off
REM Install Whisper Dictate Server as a Windows service.
REM Must be run from an elevated (Administrator) command prompt.
setlocal
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script must be run as Administrator.
    echo Right-click -^> "Run as administrator".
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo venv not found. Run setup.ps1 first.
    exit /b 1
)

REM --- Service configuration (set as machine-level env vars) ---
set MODEL=distil-large-v3
set DEVICE=cuda
set HOST=0.0.0.0
set PORT=9876
set CACHE_SIZE=2

echo Setting service environment variables (machine-wide)...
setx /M WHISPER_SERVER_MODEL      "%MODEL%"      >nul
setx /M WHISPER_SERVER_DEVICE     "%DEVICE%"     >nul
setx /M WHISPER_SERVER_HOST       "%HOST%"       >nul
setx /M WHISPER_SERVER_PORT       "%PORT%"       >nul
setx /M WHISPER_SERVER_CACHE_SIZE "%CACHE_SIZE%" >nul

echo Installing service...
"venv\Scripts\python.exe" whisper_server.py --install-service
if %errorlevel% neq 0 (
    echo Service install failed.
    exit /b %errorlevel%
)

echo Configuring auto-start...
sc.exe config WhisperDictateServer start= auto >nul

echo Starting service...
sc.exe start WhisperDictateServer

echo.
echo Done. Control the service with:
echo   sc.exe start   WhisperDictateServer
echo   sc.exe stop    WhisperDictateServer
echo   sc.exe query   WhisperDictateServer
echo   services.msc   (GUI)
echo.
echo To change config: edit the env vars at the top of this file and re-run,
echo or edit them directly in System Properties -^> Environment Variables.
echo Then restart the service.
