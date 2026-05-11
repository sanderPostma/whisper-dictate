# Whisper Dictate remote server - setup script
# Creates venv, installs dependencies (including CUDA-enabled PyTorch),
# and adds a Windows Firewall rule for the listen port.
#
# Run from an elevated PowerShell prompt:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

param(
    [int]$Port = 9876,
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=== Whisper Dictate Server Setup ===" -ForegroundColor Cyan

# 1. Locate python
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $python) { throw "Python not found on PATH. Install Python 3.10+ first." }
Write-Host "Python: $python"

# 2. Create venv
if (-not (Test-Path "$scriptDir\venv")) {
    Write-Host "Creating venv..." -ForegroundColor Yellow
    & $python -m venv venv
}

$venvPython = "$scriptDir\venv\Scripts\python.exe"
$venvPip = "$scriptDir\venv\Scripts\pip.exe"

# 3. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

# 4. Install PyTorch with CUDA FIRST (before anything that would pull CPU torch)
Write-Host "Installing PyTorch (CUDA build from $CudaIndex)..." -ForegroundColor Yellow
& $venvPip install --force-reinstall torch --index-url $CudaIndex

# 5. Install remaining deps (won't touch torch since it's already satisfied)
Write-Host "Installing whisper + server deps..." -ForegroundColor Yellow
& $venvPip install -r requirements-server.txt

# 6. Verify CUDA
Write-Host "Verifying CUDA availability..." -ForegroundColor Yellow
& $venvPython -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# 7. Firewall rule (requires admin)
$ruleName = "Whisper Dictate Server (TCP $Port)"
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Firewall rule already exists: $ruleName" -ForegroundColor Green
    } else {
        Write-Host "Adding inbound firewall rule for TCP port $Port..." -ForegroundColor Yellow
        New-NetFirewallRule -DisplayName $ruleName `
            -Direction Inbound -Protocol TCP -LocalPort $Port `
            -Action Allow -Profile Any | Out-Null
        Write-Host "Firewall rule added." -ForegroundColor Green
    }
} else {
    Write-Warning "Not running as Administrator. Skipping firewall rule."
    Write-Host "To add it manually, run in an elevated PowerShell:" -ForegroundColor Yellow
    Write-Host "  New-NetFirewallRule -DisplayName `"$ruleName`" -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Any"
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Start the server with: .\run.bat"
Write-Host "Or:                    .\venv\Scripts\python.exe whisper_server.py --device cuda --model medium.en"
