Whisper Dictate - Remote GPU server deployment
===============================================

Target: Windows machine with NVIDIA GPU at 192.168.1.87
Listen port: 9876/TCP (configurable)

Prerequisites
-------------
- Python 3.10+ (https://www.python.org/) - add to PATH during install
- NVIDIA GPU with recent drivers (CUDA 12.x compatible)
- FFmpeg on PATH (https://www.gyan.dev/ffmpeg/builds/ - "release essentials")

Install
-------
1. Extract this zip anywhere (e.g. C:\whisper-server\).
2. Open an elevated PowerShell in that folder (Right-click -> Run as Administrator).
3. Run:
       powershell -ExecutionPolicy Bypass -File .\setup.ps1
   This creates venv\, installs PyTorch (CUDA 12.1 build) + openai-whisper,
   verifies CUDA, and adds an inbound firewall rule for TCP 9876.

   If your GPU needs a different CUDA build, pass -CudaIndex:
       .\setup.ps1 -CudaIndex https://download.pytorch.org/whl/cu118

Run
---
   .\run.bat

Edit run.bat to change MODEL (tiny.en / base.en / small.en / medium.en /
large-v3 / turbo, etc.) or DEVICE (cuda / cpu).

Firewall
--------
setup.ps1 opens inbound TCP 9876 automatically when run as Administrator.
To do it manually:
   New-NetFirewallRule -DisplayName "Whisper Dictate Server (TCP 9876)" `
       -Direction Inbound -Protocol TCP -LocalPort 9876 -Action Allow -Profile Any

Client configuration (on the Linux machine)
-------------------------------------------
Edit ~/.config/whisper-dictate/config.json:
   "remote_server": {
     "enabled": true,
     "host": "192.168.1.87",
     "port": 9876,
     "model": "medium.en",
     "reconnect_check_interval_sec": 120
   }

Optional: install as a Windows service
--------------------------------------
From an elevated (Administrator) command prompt:
   install-service.bat     - register, auto-start, and start the service
   uninstall-service.bat   - stop and remove the service

The service uses these (machine-wide) environment variables, set by install-service.bat:
   WHISPER_SERVER_HOST        (default 0.0.0.0)
   WHISPER_SERVER_PORT        (default 9876)
   WHISPER_SERVER_MODEL       (default medium.en  - initial model to load)
   WHISPER_SERVER_DEVICE      (default cuda)
   WHISPER_SERVER_CACHE_SIZE  (default 2          - LRU model cache size)

To change: edit the variables at the top of install-service.bat and re-run,
then `sc.exe stop WhisperDictateServer & sc.exe start WhisperDictateServer`.
