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


_PROCEDURAL = {"tap": _tap, "felt": _felt, "brush": _brush, "breath": _breath}


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
