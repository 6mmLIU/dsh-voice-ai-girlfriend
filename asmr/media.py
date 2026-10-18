"""Real video clip library, with a daily wardrobe.

    asmr/media/outfits/<outfit>/<action>/*.mp4   one folder per outfit
    asmr/media/<action>/*.mp4                    shared fallback for any outfit

The outfit rotates by day (sorted folder names, one per day) unless the
operator pins one from the control panel. Each action folder holds
interchangeable takes of one visual moment, framed chest-down (hands +
chest, face out of frame) in the real room:

  idle/         background while nothing happens (breathing, tiny movements);
                may also hold still images (.png/.jpg) until video exists
  whisper/      played (looped) while she whispers, e.g. leaning to the mic
  ear_massage/  kneading the dummy ears      ear_pick/  ear cleaning
  ear_brush/    brushing the dummy ears      tap/       alternating taps
  felt/         rubbing the felt pad         brush/     brushing the mic
  breath/       a slow breath, chest rising  lights/ guide/  practical lights

If a clip has its own audio track (filmed with a binaural mic), that audio is
the trigger sound: picture and sound come from the same take, so they are in
sync by construction. Clips without audio get a sound from asmr/sounds/ (or
the procedural placeholder) laid over the clip's duration.

Needs ffmpeg/ffprobe on PATH to read clip audio; without it clips still play,
silently, with library sounds.
"""
from __future__ import annotations

import datetime
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
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
OUTFITS_DIR = MEDIA_DIR / "outfits"
_OUTFIT_STATE = OUT_DIR / "outfit.json"
SR = 48000
_AUDIO_CACHE = OUT_DIR / "clip-audio"


@dataclass
class Clip:
    action: str
    path: Path
    url: str
    duration: float
    audio: np.ndarray | None  # (N, 2) float32 at 48 kHz, or None
    reversed: bool = False  # a *_rev take: the forward take played backwards


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


def _load(action: str, path: Path, url: str) -> Clip | None:
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
    is_rev = path.stem.endswith("_rev")
    if is_rev and audio is None:
        # A reversed copy carries no sound of its own: play the forward
        # take's sound backwards so picture and sound still match.
        forward = path.with_name(path.stem[:-4] + path.suffix)
        fwd = _load(action, forward, url.rsplit("/", 1)[0] + "/" + forward.name) if forward.is_file() else None
        if fwd is not None and fwd.audio is not None:
            audio = np.ascontiguousarray(fwd.audio[::-1])
    clip = Clip(action, path, url, duration, audio, is_rev)
    _clips[path] = clip
    logger.info("clip %s/%s: %.2fs, %s", action, path.name, duration, "own audio" if audio is not None else "no audio")
    return clip


def outfits() -> list[str]:
    if not OUTFITS_DIR.is_dir():
        return []
    return sorted(p.name for p in OUTFITS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


def pinned_outfit() -> str | None:
    try:
        name = json.loads(_OUTFIT_STATE.read_text(encoding="utf-8")).get("pinned")
    except (OSError, ValueError):
        return None
    return name if name in outfits() else None


def pin_outfit(name: str | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUTFIT_STATE.write_text(json.dumps({"pinned": name}, ensure_ascii=False), encoding="utf-8")


def current_outfit() -> str | None:
    """Pinned outfit, else today's outfit in the rotation."""
    names = outfits()
    if not names:
        return None
    return pinned_outfit() or names[datetime.date.today().toordinal() % len(names)]


def _action_dirs(action: str) -> list[tuple[Path, str]]:
    """(folder, url prefix) to search for an action, outfit-specific first."""
    dirs = []
    outfit = current_outfit()
    if outfit:
        dirs.append((OUTFITS_DIR / outfit / action, f"/media/outfits/{outfit}/{action}"))
    dirs.append((MEDIA_DIR / action, f"/media/{action}"))
    return dirs


def _entries(action: str, exts: tuple[str, ...]) -> list[tuple[Path, str]]:
    for folder, prefix in _action_dirs(action):
        if folder.is_dir():
            found = [(p, f"{prefix}/{p.name}") for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
            if found:
                return found
    return []


def files(action: str) -> list[Path]:
    return [p for p, _ in _entries(action, VIDEO_EXTS)]


def takes(action: str) -> list[Clip]:
    """Every usable take for `action` (current outfit first), reversed copies included."""
    out = []
    for path, url in _entries(action, VIDEO_EXTS):
        clip = _load(action, path, url)
        if clip is not None:
            out.append(clip)
    return out


def pick(action: str) -> Clip | None:
    """A random video take for `action` (current outfit first), or None."""
    candidates = _entries(action, VIDEO_EXTS)
    random.shuffle(candidates)
    for path, url in candidates:
        clip = _load(action, path, url)
        if clip is not None:
            return clip
    return None


ACTIONS = ("idle", "whisper", "ear_massage", "ear_pick", "ear_brush", "tap", "felt", "brush", "breath", "lights", "guide")


def listing() -> dict:
    """What the stage needs: today's outfit, its clips, and idle stills."""
    clips = {a: [url for _, url in _entries(a, VIDEO_EXTS)] for a in ACTIONS}
    clips = {a: urls for a, urls in clips.items() if urls}
    stills = [url for _, url in _entries("idle", IMAGE_EXTS)]
    return {"outfit": current_outfit(), "clips": clips, "stills": stills}
