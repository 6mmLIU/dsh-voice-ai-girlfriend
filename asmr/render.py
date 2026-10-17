"""Render a tagged script into one 48 kHz stereo WAV plus a cue timeline.

The cue timeline is what the stage page uses to animate in sync with the
audio: which hand taps (and when), when the felt is rubbed, where the
whisper sits, when the lights alternate.
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from . import media, sounds
from .binaural import spatialize
from .config import CONFIG, OUT_DIR
from .director import parse_script, strip_tags
from .tts import synthesize

logger = logging.getLogger("asmr.render")

SR = sounds.SR
_KEEP_FILES = 60
_SIDE_AZ = {"L": -80.0, "R": 80.0, "C": 0.0}
_TRIGGER_RMS_DBFS = -27.0


@dataclass
class _Mix:
    pieces: list[tuple[int, np.ndarray]]
    cues: list[dict]
    cursor: float = 0.0  # seconds

    def add(self, stereo: np.ndarray, at: float | None = None) -> float:
        start = self.cursor if at is None else at
        self.pieces.append((int(start * SR), stereo))
        return len(stereo) / SR


def _trim_silence(x: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(x) > threshold * (np.max(np.abs(x)) or 1.0))
    if len(loud) == 0:
        return x
    pad = int(0.03 * SR)
    return x[max(0, loud[0] - pad): loud[-1] + pad]


def _distances() -> tuple[float, float]:
    audio = CONFIG.get("audio", {})
    return float(audio.get("close_distance_m", 0.12)), float(audio.get("center_distance_m", 0.25))


def _place(pos: str) -> tuple[float, float, float]:
    """(azimuth start, azimuth end, distance) for a position tag."""
    close, center = _distances()
    if pos == "SWEEP":
        return -80.0, 80.0, (close + center) / 2
    az = _SIDE_AZ.get(pos, 0.0)
    return az, az, close if pos in ("L", "R") else center


async def _speech(mix: _Mix, pos: str, text: str) -> None:
    mono, sr = await synthesize(text)
    mono = _trim_silence(sounds.resample(mono, sr))
    target = float(CONFIG.get("audio", {}).get("speech_rms_dbfs", -24))
    mono = sounds._fade(sounds.rms_normalize(mono, target))
    az0, az1, dist = _place(pos)
    dur = mix.add(spatialize(mono, SR, az0, az1, dist))
    mix.cues.append({"t": round(mix.cursor, 3), "dur": round(dur, 3), "kind": "speech", "text": text, "pos": pos})
    clip = media.pick("whisper")
    if clip is not None:
        mix.cues.append({"t": round(mix.cursor, 3), "dur": round(dur, 3), "kind": "clip", "name": "whisper",
                         "url": clip.url, "loop": True})
    mix.cursor += dur


def _taps(mix: _Mix, pos: str, fill: float | None = None) -> None:
    close, _ = _distances()
    # Hands always alternate; the current position only picks the first hand.
    count = max(2, int(fill / 0.27)) if fill else random.randint(5, 8)
    first, second = ("R", "L") if pos == "R" else ("L", "R")
    t = mix.cursor
    for i in range(count):
        side = first if i % 2 == 0 else second
        az = -70.0 if side == "L" else 70.0
        clip = sounds.get("tap")
        mix.add(spatialize(clip, SR, az, az, close + 0.03), at=t)
        mix.cues.append({"t": round(t, 3), "dur": 0.12, "kind": "sound", "name": "tap", "side": side})
        t += random.uniform(0.22, 0.32)
    mix.cursor = t + 0.1


def _felt(mix: _Mix, pos: str, fill: float | None = None) -> None:
    clip = sounds.get("felt")
    while fill and len(clip) < fill * SR:
        clip = np.concatenate([clip, sounds.get("felt")])
    if fill:
        clip = sounds._fade(clip[: int(fill * SR)].copy(), 60)
    close, center = _distances()
    if pos in ("L", "R"):
        az0 = az1 = _SIDE_AZ[pos] * 0.8
    else:
        az0, az1 = (-55.0, 55.0) if random.random() < 0.5 else (55.0, -55.0)
    dur = mix.add(spatialize(clip, SR, az0, az1, (close + center) / 2))
    mix.cues.append({"t": round(mix.cursor, 3), "dur": round(dur, 3), "kind": "sound", "name": "felt",
                     "side": pos, "az0": az0, "az1": az1})
    mix.cursor += dur


def _brush(mix: _Mix, pos: str, fill: float | None = None) -> None:
    close, _ = _distances()
    t = mix.cursor
    strokes = max(1, int(fill / 0.67)) if fill else 3
    for i in range(strokes):
        clip = sounds.get("brush")
        if pos in ("L", "R"):
            az0 = az1 = _SIDE_AZ[pos]
            direction = pos
        else:
            az0, az1 = (-75.0, 75.0) if i % 2 == 0 else (75.0, -75.0)
            direction = "LR" if i % 2 == 0 else "RL"
        dur = mix.add(spatialize(clip, SR, az0, az1, close + 0.04), at=t)
        mix.cues.append({"t": round(t, 3), "dur": round(dur, 3), "kind": "sound", "name": "brush", "side": direction})
        t += dur + 0.12
    mix.cursor = t


def _clip_audio(mix: _Mix, name: str, clip: media.Clip) -> None:
    """A clip filmed with its own (binaural) sound: use that take as-is."""
    audio = clip.audio
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms > 1e-9:
        audio = audio * (10 ** (_TRIGGER_RMS_DBFS / 20) / rms)
    dur = mix.add(audio.astype(np.float32))
    mix.cues.append({"t": round(mix.cursor, 3), "dur": round(dur, 3), "kind": "clip", "name": name, "url": clip.url})
    mix.cursor += dur


_PROCEDURAL = {"tap": _taps, "felt": _felt, "brush": _brush}


def _breath(mix: _Mix, pos: str) -> None:
    clip = sounds.get("breath")
    az0, az1, dist = _place(pos)
    dur = mix.add(spatialize(clip, SR, az0, az1, dist))
    mix.cues.append({"t": round(mix.cursor, 3), "dur": round(dur, 3), "kind": "sound", "name": "breath", "side": pos})
    mix.cursor += dur


async def render_script(script: str, meta: dict | None = None) -> dict:
    """Render `script` to asmr/out/<id>.wav and return the play event payload."""
    started = time.perf_counter()
    gap = float(CONFIG.get("audio", {}).get("gap_sec", 0.45))
    mix = _Mix(pieces=[], cues=[], cursor=0.15)
    for seg in parse_script(script):
        if seg.kind == "speech":
            await _speech(mix, seg.pos, seg.text)
        elif seg.kind == "pause":
            mix.cursor += 1.0
            continue
        elif seg.name == "lights":
            # Visual only and non-blocking: the lights alternate over whatever
            # plays next, with a short breather before it.
            clip = media.pick("lights")
            cue = {"t": round(mix.cursor, 3), "dur": 3.2, "kind": "visual", "name": "lights"}
            if clip is not None:
                cue.update(kind="clip", url=clip.url, dur=round(clip.duration, 3))
            mix.cues.append(cue)
            mix.cursor += 0.4
            continue
        else:
            clip = media.pick(seg.name)
            if clip is not None and clip.audio is not None:
                _clip_audio(mix, seg.name, clip)
            else:
                start = mix.cursor
                fill = clip.duration if clip is not None else None
                if seg.name in _PROCEDURAL:
                    _PROCEDURAL[seg.name](mix, seg.pos, fill)
                else:
                    _breath(mix, seg.pos)
                if clip is not None:
                    mix.cues.append({"t": round(start, 3), "dur": round(clip.duration, 3), "kind": "clip",
                                     "name": seg.name, "url": clip.url})
                    mix.cursor = max(mix.cursor, start + clip.duration)
        mix.cursor += gap

    end = max([mix.cursor] + [(s + len(p)) / SR for s, p in mix.pieces]) + 0.3
    out = np.zeros((int(end * SR), 2), dtype=np.float32)
    for start, piece in mix.pieces:
        out[start:start + len(piece)] += piece[: len(out) - start]
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.89:
        out *= 0.89 / peak

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    item_id = uuid.uuid4().hex[:12]
    path = OUT_DIR / f"{item_id}.wav"
    sf.write(path, out, SR, subtype="PCM_16")
    _prune()
    duration = len(out) / SR
    logger.info("rendered %s: %.2fs audio in %.2fs (%s)", item_id, duration, time.perf_counter() - started, script)
    return {
        "type": "play",
        "id": item_id,
        "url": f"/out/{item_id}.wav",
        "duration": round(duration, 3),
        "cues": mix.cues,
        "script": script,
        "display": strip_tags(script),
        **(meta or {}),
    }


def _prune() -> None:
    files = sorted(OUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    for old in files[:-_KEEP_FILES]:
        old.unlink(missing_ok=True)
