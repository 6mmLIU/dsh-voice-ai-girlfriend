"""Trigger sounds (mono, 48 kHz).

Real recordings always win: drop WAV files into asmr/sounds/<name>/ (tap,
felt, brush, breath) and one is picked at random each time. Without
recordings, a procedural placeholder is synthesized so the loop still works -
good enough to test timing and panning, but record the real thing for a
stream (a binaural mic such as a 3Dio sounds best).
"""
from __future__ import annotations

import logging
import random
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .config import SOUNDS_DIR

logger = logging.getLogger("asmr.sounds")

SR = 48000
_rng = np.random.default_rng()
_user_cache: dict[str, list[np.ndarray]] = {}


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def rms_normalize(x: np.ndarray, dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    if rms < 1e-9:
        return x
    return (x * (10 ** (dbfs / 20) / rms)).astype(np.float32)


def _fade(x: np.ndarray, ms: float = 8.0) -> np.ndarray:
    n = min(len(x) // 2, int(SR * ms / 1000))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _band(x: np.ndarray, lo: float | None, hi: float | None, order: int = 2) -> np.ndarray:
    if lo and hi:
        sos = signal.butter(order, [lo, hi], "bandpass", fs=SR, output="sos")
    elif lo:
        sos = signal.butter(order, lo, "highpass", fs=SR, output="sos")
    else:
        sos = signal.butter(order, hi, "lowpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def _tap() -> np.ndarray:
    n = int(SR * 0.14)
    t = np.arange(n) / SR
    click = _band(_rng.standard_normal(n), 1800, 7000) * np.exp(-t / 0.004)
    body_f = _rng.uniform(320, 620)
    body = np.sin(2 * np.pi * body_f * t) * np.exp(-t / 0.022) * 0.6
    return rms_normalize(_fade(click + body), -26)


def _felt() -> np.ndarray:
    dur = _rng.uniform(2.8, 3.6)
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = _band(_rng.standard_normal(n), 220, 2600, order=3)
    rate = _rng.uniform(1.2, 1.7)  # strokes per second
    phase = (t * rate) % 1.0
    stroke = np.sin(np.pi * phase) ** 1.6
    stroke *= np.repeat(_rng.uniform(0.6, 1.0, int(dur * rate) + 2), int(SR / rate) + 1)[:n]
    fibers = _band((_rng.random(n) > 0.9993).astype(float) * _rng.standard_normal(n), 3000, None) * 3.0
    return rms_normalize(_fade((noise + fibers) * stroke), -30)


def _brush() -> np.ndarray:
    n = int(SR * _rng.uniform(0.45, 0.65))
    t = np.linspace(0.0, 1.0, n)
    env = np.sin(np.pi * t**0.7) ** 2
    return rms_normalize(_fade(_band(_rng.standard_normal(n), 2800, 11000) * env), -30)


def _breath() -> np.ndarray:
    n = int(SR * _rng.uniform(1.5, 2.0))
    t = np.linspace(0.0, 1.0, n)
    env = np.sin(np.pi * t) ** 2 * (0.7 + 0.3 * np.sin(2 * np.pi * 0.8 * t))
    return rms_normalize(_fade(_band(_rng.standard_normal(n), 300, 1800) * env), -34)


def _ear_massage() -> np.ndarray:
    # Fingers kneading silicone right next to the mic capsule: muffled, slow,
    # rubbery squeezes with a soft low-mid body.
    dur = _rng.uniform(2.6, 3.4)
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = _band(_rng.standard_normal(n), 120, 1400, order=3)
    rate = _rng.uniform(0.9, 1.3)
    env = np.sin(np.pi * ((t * rate) % 1.0)) ** 2.2
    squeak = _band(_rng.standard_normal(n), 900, 2500) * (env > 0.85) * 0.4
    return rms_normalize(_fade(noise * env + squeak), -27)


def _ear_pick() -> np.ndarray:
    # Small scratchy crackles inside the ear canal, in short bursts.
    dur = _rng.uniform(2.2, 3.0)
    n = int(SR * dur)
    crackle = (_rng.random(n) > 0.997).astype(float) * _rng.standard_normal(n)
    crackle = _band(crackle, 1500, 9000) * 6.0
    scratch = _band(_rng.standard_normal(n), 2500, 8000) * 0.25
    t = np.arange(n) / SR
    bursts = (np.sin(2 * np.pi * _rng.uniform(1.5, 2.2) * t) > 0.2).astype(float)
    bursts = signal.sosfilt(signal.butter(1, 30, "lowpass", fs=SR, output="sos"), bursts)
    return rms_normalize(_fade((crackle + scratch) * bursts), -29)


def _ear_brush() -> np.ndarray:
    # A soft brush sweeping over the ear: airy swishes with a fluffy texture.
    strokes = []
    for _ in range(3):
        m = int(SR * _rng.uniform(0.5, 0.75))
        tt = np.linspace(0.0, 1.0, m)
        strokes.append(_band(_rng.standard_normal(m), 1500, 9000) * np.sin(np.pi * tt) ** 1.5)
        strokes.append(np.zeros(int(SR * 0.12)))
    return rms_normalize(_fade(np.concatenate(strokes)), -29)


def _strokes(lo: float, hi: float, n_strokes: int, stroke: tuple[float, float], gap: float, shape: float, dbfs: float) -> np.ndarray:
    parts = []
    for _ in range(n_strokes):
        m = int(SR * _rng.uniform(*stroke))
        tt = np.linspace(0.0, 1.0, m)
        parts.append(_band(_rng.standard_normal(m), lo, hi) * np.sin(np.pi * tt) ** shape * _rng.uniform(0.7, 1.0))
        parts.append(np.zeros(int(SR * _rng.uniform(gap * 0.6, gap * 1.4))))
    return rms_normalize(_fade(np.concatenate(parts)), dbfs)


def _resonant_hit(freqs: list[float], decay: float, click_band: tuple[float, float], dur: float = 0.5) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    click = _band(_rng.standard_normal(n), *click_band) * np.exp(-t / 0.003)
    body = sum(np.sin(2 * np.pi * f * _rng.uniform(0.97, 1.03) * t) * np.exp(-t / (decay / (1 + i * 0.6))) / (1 + i)
               for i, f in enumerate(freqs))
    return click * 0.6 + body


def _ear_swab() -> np.ndarray:
    return _strokes(2000, 7000, 5, (0.35, 0.55), 0.15, 1.2, -30)


def _feather() -> np.ndarray:
    return _strokes(3500, 12000, 3, (0.8, 1.2), 0.3, 2.0, -34)


def _face_brush() -> np.ndarray:
    return _strokes(1800, 10000, 3, (0.6, 0.9), 0.25, 1.6, -30)


def _wood() -> np.ndarray:
    hit = _resonant_hit([_rng.uniform(170, 260), _rng.uniform(480, 620)], 0.06, (1200, 5000), 0.4)
    return rms_normalize(_fade(hit), -25)


def _glass() -> np.ndarray:
    base = _rng.uniform(1900, 2600)
    hit = _resonant_hit([base, base * 2.32, base * 3.9], 0.35, (3000, 9000), 0.9)
    if _rng.random() < 0.35:  # a light fingernail scrape along the bottle
        m = int(SR * 0.6)
        scrape = _band(_rng.standard_normal(m), 2500, 9000) * np.sin(np.pi * np.linspace(0, 1, m)) * 0.3
        hit = np.concatenate([hit, scrape])
    return rms_normalize(_fade(hit), -27)


def _cap() -> np.ndarray:
    dur = _rng.uniform(1.4, 2.0)
    n = int(SR * dur)
    out = np.zeros(n)
    t = 0.05
    while t < dur - 0.05:
        i = int(t * SR)
        k = min(int(SR * 0.006), n - i)
        out[i:i + k] += _rng.standard_normal(k) * np.exp(-np.arange(k) / (SR * 0.0015))
        t += _rng.uniform(0.025, 0.05)
    return rms_normalize(_fade(_band(out, 1500, 8000)), -30)


def _fluffy() -> np.ndarray:
    return _strokes(150, 1600, 3, (0.7, 1.1), 0.15, 1.8, -28)


def _sponge() -> np.ndarray:
    m = int(SR * _rng.uniform(1.0, 1.4))
    tt = np.linspace(0.0, 1.0, m)
    squish = _band(_rng.standard_normal(m), 200, 2500) * np.sin(np.pi * tt) ** 0.8
    crackle = _band((_rng.random(m) > 0.995).astype(float) * _rng.standard_normal(m), 1500, 6000) * 4
    return rms_normalize(_fade(np.concatenate([squish + crackle * np.sin(np.pi * tt), np.zeros(int(SR * 0.3))])), -29)


def _velvet() -> np.ndarray:
    return _strokes(400, 3200, 2, (1.2, 1.8), 0.2, 1.4, -31)


def _page() -> np.ndarray:
    m = int(SR * _rng.uniform(0.7, 1.0))
    tt = np.linspace(0.0, 1.0, m)
    swish = _band(_rng.standard_normal(m), 800, 9000) * (tt ** 2) * (1 - tt) * 6
    flap = _resonant_hit([_rng.uniform(140, 200)], 0.03, (500, 4000), 0.2)
    return rms_normalize(_fade(np.concatenate([swish, flap, np.zeros(int(SR * _rng.uniform(1.2, 2.2)))])), -30)


def _paper() -> np.ndarray:
    dur = _rng.uniform(1.6, 2.4)
    n = int(SR * dur)
    clicks = (_rng.random(n) > 0.985).astype(float) * _rng.standard_normal(n)
    t = np.arange(n) / SR
    env = 0.4 + 0.6 * (np.sin(2 * np.pi * _rng.uniform(0.8, 1.4) * t) > 0)
    return rms_normalize(_fade(_band(clicks, 1200, 10000) * env), -29)


def _bubble() -> np.ndarray:
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    pop = _band(_rng.standard_normal(n), 1500, 8000) * np.exp(-t / 0.002) + np.sin(2 * np.pi * _rng.uniform(90, 140) * t) * np.exp(-t / 0.02) * 0.5
    return rms_normalize(_fade(pop), -24)


def _tape() -> np.ndarray:
    dur = _rng.uniform(1.2, 1.8)
    n = int(SR * dur)
    t = np.arange(n) / SR
    buzz = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * _rng.uniform(90, 160) * t))
    rasp = _band(_rng.standard_normal(n), 600, 7000) * buzz * np.minimum(1, t * 4)
    return rms_normalize(_fade(np.concatenate([rasp, np.zeros(int(SR * 0.8))])), -29)


def _gloves() -> np.ndarray:
    m = int(SR * _rng.uniform(0.5, 0.8))
    tt = np.linspace(0.0, 1.0, m)
    rub = _band(_rng.standard_normal(m), 700, 4500) * np.sin(np.pi * tt) ** 1.5
    f0 = _rng.uniform(900, 1600)
    squeak = np.sin(2 * np.pi * np.cumsum(np.full(m, f0) * (1 + 0.3 * tt)) / SR) * np.sin(np.pi * tt) ** 4 * 0.25
    return rms_normalize(_fade(np.concatenate([rub + squeak, np.zeros(int(SR * 0.2))])), -29)


def _hands() -> np.ndarray:
    return _strokes(300, 5000, 4, (0.35, 0.5), 0.05, 1.3, -30)


def _lotion() -> np.ndarray:
    m = int(SR * _rng.uniform(1.4, 2.0))
    tt = np.linspace(0.0, 1.0, m)
    rub = _band(_rng.standard_normal(m), 200, 3000) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.2 * tt * m / SR))
    wet = _band((_rng.random(m) > 0.993).astype(float) * _rng.standard_normal(m), 1000, 5000) * 5
    return rms_normalize(_fade(rub + wet), -30)


_PROCEDURAL = {
    "tap": _tap, "felt": _felt, "brush": _brush, "breath": _breath,
    "ear_massage": _ear_massage, "ear_pick": _ear_pick, "ear_brush": _ear_brush, "ear_swab": _ear_swab,
    "feather": _feather, "face_brush": _face_brush, "wood": _wood, "glass": _glass, "cap": _cap,
    "fluffy": _fluffy, "sponge": _sponge, "velvet": _velvet, "page": _page, "paper": _paper,
    "bubble": _bubble, "tape": _tape, "gloves": _gloves, "hands": _hands, "lotion": _lotion,
}


def _user_variants(name: str) -> list[np.ndarray]:
    if name in _user_cache:
        return _user_cache[name]
    variants: list[np.ndarray] = []
    folder: Path = SOUNDS_DIR / name
    for path in sorted(folder.glob("*.wav")) if folder.is_dir() else []:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            variants.append(resample(data.mean(axis=1), int(sr)))
        except Exception:  # noqa: BLE001
            logger.exception("cannot read %s", path)
    _user_cache[name] = variants
    if variants:
        logger.info("sound %r: %d recording(s) from %s", name, len(variants), folder)
    return variants


def get(name: str) -> np.ndarray:
    """One take of a trigger sound (copy, safe to modify)."""
    variants = _user_variants(name)
    if variants:
        return random.choice(variants).copy()
    return _PROCEDURAL[name]().astype(np.float32)
