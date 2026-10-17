"""Binaural placement of a mono signal around the listener's head.

Spherical-head model (Brown & Duda, 1998), no HRTF dataset needed:
  * ITD  - per-ear delay around a sphere of radius 8.75 cm (fractional,
           interpolated per sample so moving sources glide without clicks)
  * head shadow - per-ear first-order shelf: up to +6 dB treble on the ear
           facing the source, a strong treble cut on the far ear
  * near field  - inverse-distance gain per ear (softened), plus a gentle
           bass lift when the source is very close: the "whispering right in
           your ear" proximity effect

Azimuth is in degrees: -90 = left ear, 0 = straight ahead, +90 = right ear.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

HEAD_RADIUS = 0.0875  # m
SPEED_OF_SOUND = 343.0  # m/s
_ALPHA_MIN = 0.1
_THETA_MIN_DEG = 150.0
_BLOCK = 256


def _ease(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(n)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _proximity(x: np.ndarray, sr: int, dist: float) -> np.ndarray:
    amount = float(np.clip((0.3 - dist) / 0.2, 0.0, 1.0)) * 0.5
    if amount <= 0:
        return x
    sos = signal.butter(2, 250, "lowpass", fs=sr, output="sos")
    return x + amount * signal.sosfilt(sos, x)


def spatialize(
    x: np.ndarray,
    sr: int,
    az_start: float,
    az_end: float | None = None,
    dist: float = 0.2,
) -> np.ndarray:
    """Return an (N, 2) float32 stereo array placing mono `x` around the head.

    With `az_end` the source moves from `az_start` to `az_end` (eased).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = _proximity(x, sr, dist)
    az_end = az_start if az_end is None else az_end

    n_blocks = (n + _BLOCK - 1) // _BLOCK
    centers = np.minimum(np.arange(n_blocks) * _BLOCK + _BLOCK // 2, n - 1)
    az_curve = az_start + (az_end - az_start) * _ease(n)
    phi = np.radians(az_curve[centers])

    a, c = HEAD_RADIUS, SPEED_OF_SOUND
    w0 = c / a
    k = 2.0 * sr  # bilinear transform constant
    idx = np.arange(n, dtype=np.float64)
    out = np.zeros((n, 2), dtype=np.float64)

    for ch, side in enumerate((-1.0, 1.0)):  # left ear, right ear
        # Angle between the source direction and this ear's axis (0 = facing).
        theta = np.arccos(np.clip(side * np.sin(phi), -1.0, 1.0))
        theta_deg = np.degrees(theta)
        alpha = (1 + _ALPHA_MIN / 2) + (1 - _ALPHA_MIN / 2) * np.cos(
            np.radians(theta_deg / _THETA_MIN_DEG * 180.0)
        )

        # Head shadow: H(s) = (2w0 + alpha s) / (2w0 + s), block-wise, state carried.
        shadowed = np.empty(n)
        zi = np.zeros(1)
        a0 = 2 * w0 + k
        a_coef = np.array([1.0, (2 * w0 - k) / a0])
        for b in range(n_blocks):
            lo, hi = b * _BLOCK, min((b + 1) * _BLOCK, n)
            b_coef = np.array([(2 * w0 + alpha[b] * k) / a0, (2 * w0 - alpha[b] * k) / a0])
            shadowed[lo:hi], zi = signal.lfilter(b_coef, a_coef, x[lo:hi], zi=zi)

        # ITD around the sphere, shifted so the earliest ear has ~0 delay.
        itd = np.where(theta < np.pi / 2, -(a / c) * np.cos(theta), (a / c) * (theta - np.pi / 2)) + a / c
        delay = np.interp(idx, centers, itd * sr)
        delayed = np.interp(idx - delay, idx, shadowed, left=0.0, right=0.0)

        # Near-field level: inverse distance to this ear, softened so the
        # voice does not jump too much in loudness when it moves.
        px, py = dist * np.sin(phi), dist * np.cos(phi)
        d_ear = np.maximum(np.hypot(px - side * a, py), 0.03)
        gain = (np.hypot(dist, a) / d_ear) ** 0.6
        out[:, ch] = delayed * np.interp(idx, centers, gain)

    return out.astype(np.float32)
