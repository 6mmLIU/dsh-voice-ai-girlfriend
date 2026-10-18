"""Local whisper-voice TTS server for Apple Silicon (Qwen3-TTS via MLX).

    venv-tts/bin/python -m asmr.tts_server --voice whisper

POST /tts {"text": "..."} -> 24 kHz mono WAV. Every line is cloned from
asmr/voices/<voice>/ref.wav + ref.txt (made with asmr.tools.design_voice, or
an authorized real recording), so the voice stays the same across lines.
If the Base (clone) model is not available, it falls back to VoiceDesign
with the voice's instruct.txt, which keeps the style but may drift a little.

The stage server talks to it with tts.backend = "http" and
tts.http = {"url": "http://127.0.0.1:9880/tts", "json": {"text": "{text}"}}.
Runs in venv-tts (Python 3.12, mlx-audio), not venv-asmr.
"""
from __future__ import annotations

import argparse
import io
import logging
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

VOICES_DIR = Path(__file__).resolve().parent / "voices"
BASE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("asmr.tts_server")

app = FastAPI(title="ASMR whisper TTS")
_lock = threading.Lock()  # MLX generation is not re-entrant
_state: dict = {}


class TTSIn(BaseModel):
    text: str


def _load(voice: str) -> None:
    from mlx_audio.tts.utils import load_model

    folder = VOICES_DIR / voice
    ref, ref_txt = folder / "ref.wav", folder / "ref.txt"
    try:
        if not (ref.is_file() and ref_txt.is_file()):
            raise FileNotFoundError(f"{ref} / {ref_txt} missing")
        model = load_model(BASE_MODEL)
        _state.update(model=model, mode="clone", ref=str(ref), ref_text=ref_txt.read_text(encoding="utf-8").strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("clone mode unavailable (%s); using VoiceDesign", exc)
        instruct_file = folder / "candidates" / "instruct.txt"
        instruct = instruct_file.read_text(encoding="utf-8") if instruct_file.is_file() else "年轻女性温柔的耳语声，气声明显，语速很慢"
        _state.update(model=load_model(DESIGN_MODEL), mode="design", instruct=instruct)
    logger.info("voice %r ready (%s mode)", voice, _state["mode"])


@app.post("/tts")
def tts(body: TTSIn) -> Response:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    model = _state["model"]
    with _lock:
        if _state["mode"] == "clone":
            results = model.generate(text=text, ref_audio=_state["ref"], ref_text=_state["ref_text"], lang_code="chinese")
        else:
            results = model.generate(text=text, instruct=_state["instruct"], lang_code="chinese")
        audio = np.concatenate([np.asarray(r.audio) for r in results]).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, model.sample_rate, format="WAV", subtype="PCM_16")
    logger.info("%r -> %.2fs", text, len(audio) / model.sample_rate)
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/health")
def health() -> dict:
    return {"ok": "model" in _state, "mode": _state.get("mode")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default="whisper")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    args = parser.parse_args()
    _load(args.voice)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
