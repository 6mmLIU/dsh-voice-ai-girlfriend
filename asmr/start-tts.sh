#!/usr/bin/env bash
# Start the local whisper-voice TTS server (macOS: MLX, Linux: PyTorch). Extra args go to asmr.tts_server.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x venv-tts/bin/python ]; then
  python3.12 -m venv venv-tts 2>/dev/null || python3 -m venv venv-tts
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    venv-tts/bin/pip install -r asmr/requirements-tts-mlx.txt
  else
    venv-tts/bin/pip install -r asmr/requirements-tts-torch.txt
  fi
fi
exec venv-tts/bin/python -m asmr.tts_server --voice whisper "$@"
