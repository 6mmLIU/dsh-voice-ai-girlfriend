"""Load asmr-config.json (falls back to the committed example)."""
from __future__ import annotations

import json
from pathlib import Path

ASMR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASMR_DIR.parent
OUT_DIR = ASMR_DIR / "out"
SOUNDS_DIR = ASMR_DIR / "sounds"
MEDIA_DIR = ASMR_DIR / "media"
STAGE_DIR = ASMR_DIR / "stage"


def load_config() -> dict:
    for name in ("asmr-config.json", "asmr-config.example.json"):
        path = ASMR_DIR / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("asmr/asmr-config.example.json is missing")


CONFIG = load_config()
