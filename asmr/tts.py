"""Pluggable whisper TTS. Every backend returns (float32 mono samples, sample rate).

Backends (asmr-config.json -> tts.backend):
  say     macOS built-in voice. Not a whisper: a placeholder so the whole loop
          runs on a Mac with no GPU.
  bridge  this repo's bridge/voice_bridge.py  POST /api/tts {text} (OmniVoice /
          Qwen3 clone running on your GPU box).
  http    any TTS server that takes JSON and returns a WAV body, e.g.
          GPT-SoVITS api_v2 (POST /tts). "{text}" in the JSON template is
          replaced by the line to speak.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from .config import CONFIG

logger = logging.getLogger("asmr.tts")

_cache: dict[str, tuple[np.ndarray, int]] = {}
_CACHE_MAX = 200


def _decode_wav(body: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def _fill_template(value, text: str):
    if isinstance(value, str):
        return value.replace("{text}", text)
    if isinstance(value, dict):
        return {k: _fill_template(v, text) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_template(v, text) for v in value]
    return value


def _say(text: str, cfg: dict) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "say.wav"
        subprocess.run(
            [
                "say", "-v", str(cfg.get("voice", "Tingting")), "-r", str(cfg.get("rate", 135)),
                "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(out), text,
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        data, sr = sf.read(out, dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


async def _bridge(text: str, cfg: dict) -> tuple[np.ndarray, int]:
    async with httpx.AsyncClient(timeout=cfg.get("timeout_sec", 60)) as client:
        resp = await client.post(cfg.get("base", "http://127.0.0.1:8765").rstrip("/") + "/api/tts", json={"text": text})
        resp.raise_for_status()
        return _decode_wav(resp.content)


async def _http(text: str, cfg: dict) -> tuple[np.ndarray, int]:
    body = _fill_template(copy.deepcopy(cfg.get("json", {"text": "{text}"})), text)
    async with httpx.AsyncClient(timeout=cfg.get("timeout_sec", 60)) as client:
        if str(cfg.get("method", "POST")).upper() == "GET":
            resp = await client.get(cfg["url"], params=body)
        else:
            resp = await client.post(cfg["url"], json=body)
        resp.raise_for_status()
        return _decode_wav(resp.content)


async def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Speak one short line. Results are cached (idle lines repeat a lot)."""
    tts_cfg = CONFIG.get("tts", {})
    backend = tts_cfg.get("backend", "say")
    key = hashlib.sha1(f"{backend}|{text}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    cfg = tts_cfg.get(backend, {})
    if backend == "say":
        result = await asyncio.to_thread(_say, text, cfg)
    elif backend == "bridge":
        result = await _bridge(text, cfg)
    elif backend == "http":
        result = await _http(text, cfg)
    else:
        raise ValueError(f"unknown tts.backend: {backend!r}")
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = result
    logger.info("TTS[%s] %r -> %.2fs", backend, text, len(result[0]) / result[1])
    return result
