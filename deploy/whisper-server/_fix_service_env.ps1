$pyHome     = 'C:\Users\natty\AppData\Local\Python\pythoncore-3.13-64'
$venvSite   = 'C:\sander\whisper-server\venv\Lib\site-packages'
$win32Dir   = "$venvSite\win32"
$win32Lib   = "$venvSite\win32\lib"
$pywinSys32 = "$venvSite\pywin32_system32"
$key        = 'HKLM:\SYSTEM\CurrentControlSet\Services\WhisperDictateServer'

$env = @(
    "PATH=$pyHome;$pywinSys32;C:\Windows\system32;C:\Windows",
    "PYTHONPATH=$venvSite;$win32Dir;$win32Lib"
)
Set-ItemProperty -Path $key -Name Environment -Value $env -Type MultiString
(Get-ItemProperty -Path $key -Name Environment).Environment | ForEach-Object { Write-Host "  $_" }
