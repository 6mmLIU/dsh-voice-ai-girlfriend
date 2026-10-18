#!/usr/bin/env bash
# Forward WeChat Channels (视频号) live comments to the stage (macOS / Linux). Extra args: --dump, --dom.
# On a Linux server without a desktop, run it under a virtual display: xvfb-run ./asmr/start-channels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
venv-asmr/bin/python -m playwright install chromium >/dev/null
exec venv-asmr/bin/python -m asmr.danmaku.wechat_channels "$@"
