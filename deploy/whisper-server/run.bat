@echo off
REM Whisper Dictate remote server launcher
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo venv not found. Run setup.ps1 first.
    exit /b 1
)

set HOST=0.0.0.0
set PORT=9876
set MODEL=distil-large-v3
set DEVICE=cuda

echo Starting Whisper server on %HOST%:%PORT% (model=%MODEL%, device=%DEVICE%)
"venv\Scripts\python.exe" whisper_server.py --host %HOST% --port %PORT% --model %MODEL% --device %DEVICE%
