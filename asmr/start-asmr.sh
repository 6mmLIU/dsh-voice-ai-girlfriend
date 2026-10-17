#!/usr/bin/env bash
# Start the ASMR stage server from the repo root's venv-asmr.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x venv-asmr/bin/python ]; then
  python3 -m venv venv-asmr
  venv-asmr/bin/pip install -r asmr/requirements.txt
fi
exec venv-asmr/bin/python -m asmr.server
