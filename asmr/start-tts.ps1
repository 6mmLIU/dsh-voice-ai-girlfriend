# Start the local whisper-voice TTS server on Windows (PyTorch engine).
#   powershell -ExecutionPolicy Bypass -File asmr\start-tts.ps1            # NVIDIA GPU or CPU
#   powershell -ExecutionPolicy Bypass -File asmr\start-tts.ps1 --small    # 0.6B model, faster on CPU
# For an NVIDIA GPU install the CUDA build of PyTorch into venv-tts first
# (see asmr\requirements-tts-torch.txt).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "venv-tts\Scripts\python.exe")) {
  py -3.12 -m venv venv-tts
  & venv-tts\Scripts\python.exe -m pip install -r asmr\requirements-tts-torch.txt
}
& venv-tts\Scripts\python.exe -m asmr.tts_server --voice whisper @args
