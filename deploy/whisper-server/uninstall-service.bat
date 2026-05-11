@echo off
REM Uninstall Whisper Dictate Server Windows service.
setlocal
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script must be run as Administrator.
    pause
    exit /b 1
)

echo Stopping service (if running)...
sc.exe stop WhisperDictateServer >nul 2>&1

echo Removing service...
"venv\Scripts\python.exe" whisper_server.py --remove-service

echo Clearing env vars...
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v WHISPER_SERVER_MODEL      /f >nul 2>&1
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v WHISPER_SERVER_DEVICE     /f >nul 2>&1
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v WHISPER_SERVER_HOST       /f >nul 2>&1
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v WHISPER_SERVER_PORT       /f >nul 2>&1
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v WHISPER_SERVER_CACHE_SIZE /f >nul 2>&1

echo Done.
