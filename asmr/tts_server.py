"""Local whisper-voice TTS server (Qwen3-TTS) for macOS, Windows and Linux.

    venv-tts/bin/python -m asmr.tts_server --voice whisper          (macOS / Linux)
    venv-tts\\Scripts\\python -m asmr.tts_server --voice whisper     (Windows)

POST /tts {"text": "..."} -> mono WAV. Every line is cloned from
asmr/voices/<voice>/ref.wav + ref.txt (made with asmr.tools.design_voice, or
an authorized real recording), so the voice stays the same across lines.
If the clone model is not available, it falls back to VoiceDesign with the
voice's instruct.txt, which keeps the style but may drift a little.

Engine (asmr/tts_engines.py): MLX on Apple Silicon, PyTorch elsewhere (NVIDIA
GPU if present, else CPU; --small uses the 0.6B clone model, much faster on CPU).

The stage server talks to it with tts.backend = "local"
(tts.local.url = "http://127.0.0.1:9880/tts").
"""
from __future__ import annotations

import argparse
import io
import logging
import threading
from pathlib import Path

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import tts_engines

VOICES_DIR = Path(__file__).resolve().parent / "voices"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("asmr.tts_server")

app = FastAPI(title="ASMR whisper TTS")
_lock = threading.Lock()  # generation is not re-entrant
_state: dict = {}


class TTSIn(BaseModel):
    text: str


def _load(voice: str, engine: str, small: bool) -> None:
    folder = VOICES_DIR / voice
    ref, ref_txt = folder / "ref.wav", folder / "ref.txt"
    try:
        if not (ref.is_file() and ref_txt.is_file()):
            raise FileNotFoundError(f"{ref} / {ref_txt} missing")
        model = tts_engines.load("base", engine, small)
        _state.update(model=model, mode="clone", ref=str(ref), ref_text=ref_txt.read_text(encoding="utf-8").strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("clone mode unavailable (%s); using VoiceDesign", exc)
        instruct_file = folder / "candidates" / "instruct.txt"
        instruct = instruct_file.read_text(encoding="utf-8") if instruct_file.is_file() else "年轻女性温柔的耳语声，气声明显，语速很慢"
        _state.update(model=tts_engines.load("design", engine), mode="design", instruct=instruct)
    logger.info("voice %r ready (%s mode, %s engine)", voice, _state["mode"], _state["model"].name)


@app.post("/tts")
def tts(body: TTSIn) -> Response:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    model = _state["model"]
    with _lock:
        if _state["mode"] == "clone":
            audio = model.clone(text, _state["ref"], _state["ref_text"])
        else:
            audio = model.design(text, _state["instruct"])
    buf = io.BytesIO()
    sf.write(buf, audio, model.sample_rate, format="WAV", subtype="PCM_16")
    logger.info("%r -> %.2fs", text, len(audio) / model.sample_rate)
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/health")
def health() -> dict:
    model = _state.get("model")
    return {"ok": model is not None, "mode": _state.get("mode"), "engine": model and model.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default="whisper")
    parser.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch"])
    parser.add_argument("--small", action="store_true", help="0.6B clone model (torch engine; faster on CPU)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    args = parser.parse_args()
    _load(args.voice, args.engine, args.small)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
