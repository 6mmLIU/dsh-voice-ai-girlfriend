"""Real video clip library: asmr/media/<action>/*.mp4

Each action folder holds interchangeable takes of one visual moment, framed
chest-down (hands + chest, face out of frame) in the real room:

  idle/     background loop while nothing happens (breathing, tiny movements)
  whisper/  played (looped) while she whispers, e.g. leaning towards the mic
  tap/      fingertips tapping, hands alternating
  felt/     rubbing the felt pad
  brush/    brushing the mic / felt
  breath/   a slow breath, chest rising
  lights/   the practical lamps alternating left/right

If a clip has its own audio track (filmed with a binaural mic), that audio is
the trigger sound: picture and sound come from the same take, so they are in
sync by construction. Clips without audio get a sound from asmr/sounds/ (or
the procedural placeholder) laid over the clip's duration.

Needs ffmpeg/ffprobe on PATH to read clip audio; without it clips still play,
silently, with library sounds.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MEDIA_DIR, OUT_DIR

logger = logging.getLogger("asmr.media")

VIDEO_EXTS = (".mp4", ".webm", ".mov")
SR = 48000
_AUDIO_CACHE = OUT_DIR / "clip-audio"


@dataclass
class Clip:
    action: str
    path: Path
    url: str
    duration: float
    audio: np.ndarray | None  # (N, 2) float32 at 48 kHz, or None


_clips: dict[Path, Clip] = {}


def _probe(path: Path) -> tuple[float, bool]:
    if not shutil.which("ffprobe"):
        return 4.0, False
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    info = json.loads(out.stdout)
    duration = float(info.get("format", {}).get("duration") or 4.0)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return duration, has_audio


def _extract_audio(path: Path) -> np.ndarray | None:
    stat = path.stat()
    key = hashlib.sha1(f"{path}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()[:16]
    wav = _AUDIO_CACHE / f"{key}.wav"
    if not wav.is_file():
        _AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "2", "-ar", str(SR), str(wav)],
            capture_output=True, timeout=120, check=True,
        )
    data, _ = sf.read(wav, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2]


def _load(action: str, path: Path) -> Clip | None:
    cached = _clips.get(path)
    if cached is not None and cached.path.stat().st_mtime_ns == path.stat().st_mtime_ns:
        return cached
    try:
        duration, has_audio = _probe(path)
        audio = _extract_audio(path) if has_audio and shutil.which("ffmpeg") else None
    except Exception:  # noqa: BLE001
        logger.exception("cannot read clip %s", path)
        return None
    if audio is not None and float(np.max(np.abs(audio))) < 1e-4:
        audio = None  # silent track: treat as no audio
    clip = Clip(action, path, f"/media/{action}/{path.name}", duration, audio)
    _clips[path] = clip
    logger.info("clip %s/%s: %.2fs, %s", action, path.name, duration, "own audio" if audio is not None else "no audio")
    return clip


def files(action: str) -> list[Path]:
    folder = MEDIA_DIR / action
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def pick(action: str) -> Clip | None:
    """A random take for `action`, or None when the folder has no clips."""
    candidates = files(action)
    random.shuffle(candidates)
    for path in candidates:
        clip = _load(action, path)
        if clip is not None:
            return clip
    return None


def listing() -> dict[str, list[str]]:
    """{action: [url, ...]} for the stage page."""
    out: dict[str, list[str]] = {}
    if MEDIA_DIR.is_dir():
        for folder in sorted(p for p in MEDIA_DIR.iterdir() if p.is_dir()):
            names = [f"/media/{folder.name}/{p.name}" for p in files(folder.name)]
            if names:
                out[folder.name] = names
    return out
