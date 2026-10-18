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

from . import actions, media, sounds
from .binaural import spatialize
from .config import CONFIG, OUT_DIR
from .director import parse_script, strip_tags
from .tts import synthesize

logger = logging.getLogger("asmr.render")

SR = sounds.SR
_KEEP_FILES = 60
_SIDE_AZ = {"L": -80.0, "R": 80.0, "C": 0.0}
_TRIGGER_RMS_DBFS = -27.0
_TRANSITION_RMS_DBFS = -32.0
_RATE_RANGE = (0.9, 1.1)  # per-take speed variation so repeats don't read as a loop
_NO_REVERSE = {"tap", "wood", "glass", "cap", "bubble", "tape", "page"}  # setting props down: quieter than the trigger itself
_EAR_TURN_SEC = 6.0  # how long to stay on one dummy ear before switching


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
    _whisper_takes(mix, mix.cursor, dur)
    mix.cursor += dur


def _whisper_takes(mix: _Mix, start: float, dur: float) -> None:
    """Lean-in takes covering a spoken line. A take is never cut mid-move (that
    would jump back to the idle frame): whole takes are sped up or slowed down
    to fill the line, and a line too short for a whole take gets none."""
    clip = media.pick("whisper")
    if clip is None:
        return
    n = max(1, round(dur / clip.duration))
    rate = n * clip.duration / dur
    if not 0.75 <= rate <= 1.3:
        return
    t, last_url = start, None
    for _ in range(n):
        take = _next_take("whisper", last_url) or clip
        rate_i = take.duration / (dur / n)
        mix.cues.append({"t": round(t, 3), "dur": round(dur / n, 3), "kind": "clip", "name": "whisper",
                         "url": take.url, "rate": round(rate_i, 3)})
        t += dur / n
        last_url = take.url


def _loop(key: str, seconds: float) -> np.ndarray:
    """Takes of a trigger sound back to back until `seconds` long."""
    parts, total = [], 0
    while total < seconds * SR:
        take = sounds.get(key)
        parts.append(take)
        total += len(take)
    return sounds._fade(np.concatenate(parts)[: int(seconds * SR)].copy(), 60)


def _sweep_curve(n: int, pos: str) -> np.ndarray:
    """A slow drift from ear to ear and back (or within one side)."""
    t = np.arange(n) / SR
    if pos in ("L", "R"):
        sign = -1.0 if pos == "L" else 1.0
        return sign * (57.5 + 27.5 * np.sin(2 * np.pi * t / 7.0))
    return 70.0 * np.sin(2 * np.pi * t / 8.0 + random.uniform(0, 2 * np.pi))


def _procedural(mix: _Mix, key: str, pos: str, start: float, seconds: float) -> None:
    """Lay `seconds` of the action's sound (recordings or placeholder) from `start`."""
    close, center = _distances()
    placement = actions.BY_KEY[key].placement if key in actions.BY_KEY else "center"
    end = start + seconds
    if placement == "ear":
        # Stay on one dummy ear for a while, then switch to the other one
        # (unless the script named a side).
        t, side = start, (pos if pos in ("L", "R") else random.choice("LR"))
        while t < end - 0.5:
            span = min(_EAR_TURN_SEC, end - t)
            az = -88.0 if side == "L" else 88.0
            mix.add(spatialize(_loop(key, span), SR, az, az, 0.06), at=t)
            mix.cues.append({"t": round(t, 3), "dur": round(span, 3), "kind": "sound", "name": key, "side": side})
            t += span
            if pos not in ("L", "R"):
                side = "R" if side == "L" else "L"
        return
    if placement == "alternate":
        # Slow, separate hits, alternating left and right hands.
        first, second = ("R", "L") if pos == "R" else ("L", "R")
        t, i = start, 0
        while t < end - 0.3:
            take = sounds.get(key)
            side = first if i % 2 == 0 else second
            az = -70.0 if side == "L" else 70.0
            mix.add(spatialize(take, SR, az, az, close + 0.03), at=t)
            mix.cues.append({"t": round(t, 3), "dur": round(len(take) / SR, 3), "kind": "sound", "name": key, "side": side})
            t += len(take) / SR + random.uniform(0.35, 0.75)
            i += 1
        return
    audio = _loop(key, seconds)
    if placement == "sweep":
        curve, dist = _sweep_curve(len(audio), pos), (close + center) / 2
    else:  # center: in front of the mic, drifting a little
        curve = _SIDE_AZ.get(pos, 0.0) * 0.75 + 12.0 * np.sin(2 * np.pi * np.arange(len(audio)) / SR / 9.0)
        dist = close + 0.05
    mix.add(spatialize(audio, SR, 0.0, None, dist, az_curve=curve), at=start)
    mix.cues.append({"t": round(start, 3), "dur": round(seconds, 3), "kind": "sound", "name": key, "side": pos})


def _stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Play `audio` at `rate` (like a video playbackRate): shorter and a touch higher when > 1."""
    if abs(rate - 1.0) < 1e-3:
        return audio
    n = int(len(audio) / rate)
    src = np.arange(n) * rate
    idx = np.arange(len(audio))
    return np.stack([np.interp(src, idx, audio[:, c]) for c in range(audio.shape[1])], axis=1).astype(np.float32)


def _take_audio(clip: media.Clip, rate: float, dbfs: float) -> np.ndarray | None:
    if clip.audio is None:
        return None
    audio = clip.audio
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms > 1e-9:
        audio = audio * (10 ** (dbfs / 20) / rms)
    return _stretch(audio.astype(np.float32), rate)


def _transition(mix: _Mix, name: str) -> None:
    """A scene transition clip (props set down / cleared) with its own sound."""
    clip = media.pick(name)
    if clip is None:
        return
    audio = _take_audio(clip, 1.0, _TRANSITION_RMS_DBFS)
    if audio is not None:
        mix.add(audio)
    mix.cues.append({"t": round(mix.cursor, 3), "dur": round(clip.duration, 3), "kind": "clip", "name": name, "url": clip.url})
    mix.cursor += clip.duration


def _next_take(key: str, last_url: str | None) -> media.Clip | None:
    """A take that is not the one just played (when there is a choice).
    Percussive actions skip reversed takes: a backwards tap looks and sounds wrong."""
    pool = media.takes(key)
    if key in _NO_REVERSE or (key in actions.BY_KEY and actions.BY_KEY[key].placement == "alternate"):
        pool = [c for c in pool if not c.reversed] or pool
    fresh = [c for c in pool if c.url != last_url]
    return random.choice(fresh or pool) if pool else None


def _action(mix: _Mix, key: str, pos: str, seconds: float) -> None:
    """Perform one action for `seconds`: chain video takes (their own sound,
    or library sound laid over them) or, with no footage, sound alone.

    Takes never repeat back to back when there is more than one, and each
    plays at a slightly different speed, so a long action does not read as a
    loop. Prop actions are wrapped in their scene's set-down / clear-away clips."""
    scene = actions.BY_KEY[key].scene if key in actions.BY_KEY else ""
    if scene and media.files(key):
        _transition(mix, f"scene_{scene}_in")
    start = t = mix.cursor
    takes, last_url = 0, None
    while t < start + seconds - 1.0:
        clip = _next_take(key, last_url)
        if clip is None:
            break
        rate = round(random.uniform(*_RATE_RANGE), 3)
        dur = clip.duration / rate
        audio = _take_audio(clip, rate, _TRIGGER_RMS_DBFS)
        if audio is not None:
            mix.add(audio, at=t)
        else:
            side = pos if pos in ("L", "R") else ("L" if takes % 2 == 0 else "R")
            _procedural(mix, key, side, t, dur)
        mix.cues.append({"t": round(t, 3), "dur": round(dur, 3), "kind": "clip", "name": key, "url": clip.url, "rate": rate})
        t += dur
        takes += 1
        last_url = clip.url
    if takes == 0:
        _procedural(mix, key, pos, start, seconds)
        t = start + seconds
    mix.cursor = t
    if scene and takes:
        _transition(mix, f"scene_{scene}_out")


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
        elif seg.name in ("lights", "guide"):
            # Visual only and non-blocking: the light plays over whatever comes
            # next. "guide" is a slow breathing light (4 s in, 6 s out) to follow.
            dur = seg.seconds or (20.0 if seg.name == "guide" else 8.0)
            mix.cues.append({"t": round(mix.cursor, 3), "dur": dur, "kind": "visual", "name": seg.name})
            clip = media.pick(seg.name)
            if clip is not None:  # a whole take, never cut mid-move
                mix.cues.append({"t": round(mix.cursor, 3), "dur": round(clip.duration, 3), "kind": "clip",
                                 "name": seg.name, "url": clip.url})
            mix.cursor += 0.4
            continue
        else:
            _action(mix, seg.name, seg.pos, seg.seconds or actions.seconds(seg.name))
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
