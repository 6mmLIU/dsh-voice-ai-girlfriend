"""Qwen3-TTS on any computer: MLX on Apple Silicon, PyTorch everywhere else.

Both engines run the same Qwen3-TTS models (Base = voice clone, VoiceDesign =
voice from a text description) and return (float32 mono samples, sample rate).

  mlx    Apple Silicon Macs (mlx-audio, 8-bit models)
  torch  Windows / Linux / Intel Macs (qwen-tts). Uses an NVIDIA GPU when one
         is available (CUDA, bfloat16), otherwise the CPU (float32, slow -
         the 0.6B clone model helps there).
"""
from __future__ import annotations

import platform
import sys

import numpy as np

MLX_MODELS = {
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
}
TORCH_MODELS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def pick_engine(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_audio  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "torch"


class MLXEngine:
    name = "mlx"

    def __init__(self, kind: str) -> None:
        from mlx_audio.tts.utils import load_model

        self.model = load_model(MLX_MODELS["design" if kind == "design" else "base"])
        self.sample_rate = self.model.sample_rate

    def clone(self, text: str, ref_audio: str, ref_text: str) -> np.ndarray:
        results = self.model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, lang_code="chinese")
        return np.concatenate([np.asarray(r.audio) for r in results]).astype(np.float32)

    def design(self, text: str, instruct: str) -> np.ndarray:
        results = self.model.generate(text=text, instruct=instruct, lang_code="chinese")
        return np.concatenate([np.asarray(r.audio) for r in results]).astype(np.float32)


class TorchEngine:
    name = "torch"

    def __init__(self, kind: str, small: bool = False) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        cuda = torch.cuda.is_available()
        key = "design" if kind == "design" else ("base-0.6b" if small else "base")
        self.model = Qwen3TTSModel.from_pretrained(
            TORCH_MODELS[key],
            device_map="cuda:0" if cuda else "cpu",
            dtype=torch.bfloat16 if cuda else torch.float32,
            attn_implementation="sdpa",
        )
        self.device = "cuda" if cuda else "cpu"
        self.sample_rate = None
        self._prompt = None

    def clone(self, text: str, ref_audio: str, ref_text: str) -> np.ndarray:
        if self._prompt is None:  # the reference voice is encoded once, then reused
            self._prompt = self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text,
                                                                x_vector_only_mode=False)
        wavs, sr = self.model.generate_voice_clone(text=text, language="Chinese", voice_clone_prompt=self._prompt)
        self.sample_rate = sr
        return np.asarray(wavs[0], dtype=np.float32)

    def design(self, text: str, instruct: str) -> np.ndarray:
        wavs, sr = self.model.generate_voice_design(text=text, language="Chinese", instruct=instruct)
        self.sample_rate = sr
        return np.asarray(wavs[0], dtype=np.float32)


def load(kind: str, engine: str = "auto", small: bool = False):
    """kind: "base" (clone) or "design"."""
    engine = pick_engine(engine)
    return MLXEngine(kind) if engine == "mlx" else TorchEngine(kind, small)
