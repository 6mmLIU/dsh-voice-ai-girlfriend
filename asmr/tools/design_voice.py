"""Design the stream's whisper voice from a text description (no recording needed).

Runs Qwen3-TTS VoiceDesign locally (MLX on Apple Silicon, PyTorch on Windows /
Linux, see asmr/tts_engines.py) and writes a few candidate takes. Listen, keep the one you like as the voice's reference
(ref.wav + ref.txt); the local TTS server then clones it for every line so
the voice stays the same all stream long.

    venv-tts/bin/python -m asmr.tools.design_voice --name whisper --takes 3
    venv-tts/bin/python -m asmr.tools.design_voice --name whisper --keep 2

Setup (Python 3.12 venv, separate from venv-asmr):
    macOS (Apple Silicon): pip install -r asmr/requirements-tts-mlx.txt
    Windows / Linux:       pip install -r asmr/requirements-tts-torch.txt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import soundfile as sf

VOICES_DIR = Path(__file__).resolve().parents[1] / "voices"

DEFAULT_INSTRUCT = (
    "年轻女性的耳语声，音色温柔甜美，带明显的气声，音量很轻，语速很慢，"
    "像贴在耳边轻声说悄悄话，放松、亲密、适合助眠"
)
DEFAULT_TEXT = "晚上好呀…今天也辛苦了吧…慢慢放松，我会一直在这里陪着你哦。"


def design(name: str, takes: int, instruct: str, text: str, engine: str = "auto") -> None:
    from .. import tts_engines

    out = VOICES_DIR / name / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    model = tts_engines.load("design", engine)
    for i in range(1, takes + 1):
        audio = model.design(text, instruct)
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
    parser.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch"])
    args = parser.parse_args()
    if args.keep:
        keep(args.name, args.keep)
    else:
        design(args.name, args.takes, args.instruct, args.text, args.engine)


if __name__ == "__main__":
    main()
