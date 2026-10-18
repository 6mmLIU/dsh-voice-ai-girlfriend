"""Normalize raw footage into the stage's clip library.

    venv-asmr/bin/python -m asmr.tools.prepare_clips raw/tap_01.MOV --action tap
    venv-asmr/bin/python -m asmr.tools.prepare_clips raw/idle.mp4 --action idle --trim 3.5 18
    venv-asmr/bin/python -m asmr.tools.prepare_clips raw/felt_*.mp4 --action felt --no-audio
    venv-asmr/bin/python -m asmr.tools.prepare_clips h3/ear_massage.mp4 --action ear_massage --outfit 1-lavender-knit

Every clip becomes 1080x1920 (9:16, center-cropped), 30 fps, H.264 with
faststart so the browser can start it instantly, and 48 kHz stereo AAC when
the take has sound worth keeping (filmed with a binaural mic: that sound is
used as the trigger sound, in sync with the picture).

Output: asmr/media/<action>/<name>.mp4, or with --outfit
asmr/media/outfits/<outfit>/<action>/<name>.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import MEDIA_DIR

ACTIONS = ("idle", "whisper", "ear_massage", "ear_pick", "ear_brush", "tap", "wood", "glass", "cap", "fluffy", "sponge",
           "velvet", "felt", "brush", "breath", "lights", "guide", "scene_<name>_in", "scene_<name>_out")


def prepare(src: Path, action: str, keep_audio: bool, trim: tuple[float, float] | None, width: int, height: int,
            outfit: str | None = None) -> Path:
    out_dir = (MEDIA_DIR / "outfits" / outfit / action) if outfit else (MEDIA_DIR / action)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if trim:
        cmd += ["-ss", str(trim[0]), "-to", str(trim[1])]
    cmd += ["-i", str(src)]
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=30,format=yuv420p"
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-movflags", "+faststart"]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)
    return dst


def make_reverse(clip: Path) -> Path:
    """A backwards copy (picture only). The stage pairs it with the forward
    take's sound played backwards, giving a long action a second, different
    take for free. Only for smooth gestures (rubbing, stroking), not taps."""
    dst = clip.with_name(f"{clip.stem}_rev.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", "reverse", "-an", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "19", "-movflags", "+faststart", str(dst)], check=True)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="raw video files")
    parser.add_argument("--action", required=True, help=f"clip folder, usually one of: {', '.join(ACTIONS)}")
    parser.add_argument("--no-audio", action="store_true", help="drop the clip's sound (library sounds are used instead)")
    parser.add_argument("--trim", nargs=2, type=float, metavar=("START", "END"), help="keep only START..END seconds")
    parser.add_argument("--size", default="1080x1920", help="output WxH, default 1080x1920")
    parser.add_argument("--outfit", help="file it under asmr/media/outfits/<outfit>/ (daily wardrobe)")
    parser.add_argument("--reverse", action="store_true", help="also make a backwards take (smooth gestures only)")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (macOS: brew install ffmpeg; Windows: winget install ffmpeg)")
    width, height = (int(v) for v in args.size.lower().split("x"))
    for src in args.inputs:
        dst = prepare(src, args.action, not args.no_audio, tuple(args.trim) if args.trim else None, width, height, args.outfit)
        print(f"{src} -> {dst}")
        if args.reverse:
            print(f"  + {make_reverse(dst)}")


if __name__ == "__main__":
    main()
