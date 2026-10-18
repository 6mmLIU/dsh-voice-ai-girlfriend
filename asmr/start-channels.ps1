# Forward WeChat Channels (视频号) live comments to the stage (Windows). Extra args: --dump, --dom.
#   powershell -ExecutionPolicy Bypass -File asmr\start-channels.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
& venv-asmr\Scripts\python.exe -m playwright install chromium | Out-Null
& venv-asmr\Scripts\python.exe -m asmr.danmaku.wechat_channels @args
