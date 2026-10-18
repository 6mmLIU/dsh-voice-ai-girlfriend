"""Design the stream's whisper voice from a text description (no recording needed).

Runs Qwen3-TTS VoiceDesign locally on Apple Silicon (MLX) and writes a few
candidate takes. Listen, keep the one you like as the voice's reference
(ref.wav + ref.txt); the local TTS server then clones it for every line so
the voice stays the same all stream long.

    venv-tts/bin/python -m asmr.tools.design_voice --name whisper --takes 3
    venv-tts/bin/python -m asmr.tools.design_voice --name whisper --keep 2

Setup (Python 3.12 venv, separate from venv-asmr):
    uv venv venv-tts --python 3.12 && uv pip install --python venv-tts/bin/python mlx-audio fastapi uvicorn soundfile
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

VOICES_DIR = Path(__file__).resolve().parents[1] / "voices"
DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

DEFAULT_INSTRUCT = (
    "年轻女性的耳语声，音色温柔甜美，带明显的气声，音量很轻，语速很慢，"
    "像贴在耳边轻声说悄悄话，放松、亲密、适合助眠"
)
DEFAULT_TEXT = "晚上好呀…今天也辛苦了吧…慢慢放松，我会一直在这里陪着你哦。"


def design(name: str, takes: int, instruct: str, text: str) -> None:
    from mlx_audio.tts.utils import load_model

    out = VOICES_DIR / name / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(DESIGN_MODEL)
    for i in range(1, takes + 1):
        chunks = [np.asarray(r.audio) for r in model.generate(text=text, instruct=instruct, lang_code="chinese")]
        audio = np.concatenate(chunks).astype(np.float32)
        path = out / f"take{i}.wav"
        sf.write(path, audio, model.sample_rate)
        print(f"{path}  {len(audio) / model.sample_rate:.1f}s")
    (out / "text.txt").write_text(text, encoding="utf-8")
    (out / "instruct.txt").write_text(instruct, encoding="utf-8")


def keep(name: str, take: int) -> None:
    folder = VOICES_DIR / name
    src = folder / "candidates" / f"take{take}.wav"
    shutil.copy(src, folder / "ref.wav")
    shutil.copy(folder / "candidates" / "text.txt", folder / "ref.txt")
    print(f"voice {name!r} now uses {src.name} as its reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="whisper", help="voice folder under asmr/voices/")
    parser.add_argument("--takes", type=int, default=3)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT, help="voice description")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="what the reference take says")
    parser.add_argument("--keep", type=int, help="use candidate N as the voice reference")
    args = parser.parse_args()
    if args.keep:
        keep(args.name, args.keep)
    else:
        design(args.name, args.takes, args.instruct, args.text)


if __name__ == "__main__":
    main()
