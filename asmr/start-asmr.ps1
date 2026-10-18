# Start the ASMR stage server (Windows).
#   powershell -ExecutionPolicy Bypass -File asmr\start-asmr.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "venv-asmr\Scripts\python.exe")) {
  py -3 -m venv venv-asmr
  & venv-asmr\Scripts\python.exe -m pip install -r asmr\requirements.txt
}
& venv-asmr\Scripts\python.exe -m asmr.server
