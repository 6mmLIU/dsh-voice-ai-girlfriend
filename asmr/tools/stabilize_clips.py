"""Make an outfit's clips seam-perfect: remove zoom drift, pin boundary frames.

AI video (H3) creeps the camera in by ~1% over a clip even when asked for a
locked-off shot, so a clip's last frame is slightly bigger than the idle
frame it cuts back to: on stream the picture visibly "shrinks" at every seam.

For every clip of the outfit this tool
  1. measures the zoom of each frame against the clip's first frame (on the
     background only, where hands don't move) and fits a smooth drift curve,
  2. re-zooms every frame so the whole clip sits at one fixed framing,
  3. blends the first and last few frames into one canonical frame per state
     ("base" = the idle frame, or a prop scene's frame), so every cut between
     clips happens between pixel-identical frames.
Originals are kept in <outfit>/.orig/ and always used as the source, so the
tool can be re-run safely. Reversed takes (*_rev) are rebuilt afterwards.

    venv-asmr/bin/python -m asmr.tools.stabilize_clips --outfit 1-lavender-knit
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .. import actions
from ..config import MEDIA_DIR
from .prepare_clips import make_reverse

LOW_W, LOW_H = 270, 480
SCALES = np.arange(0.990, 1.0301, 0.001)
BLEND = 8  # frames blended into the canonical boundary frame at each end


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": num / den,
            "audio": any(s["codec_type"] == "audio" for s in streams)}


def _frames(path: Path, w: int, h: int):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def _zoom(img: np.ndarray, z: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.array([[z, 0, (1 - z) * w / 2], [0, z, (1 - z) * h / 2]], dtype=np.float32)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _low(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(img, (LOW_W, LOW_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


_MASK = np.zeros((LOW_H, LOW_W), bool)
_MASK[: int(LOW_H * 0.22)] = True  # room behind her
_MASK[int(LOW_H * 0.82):] = True  # desk edge / foreground
_MASK[:, :12] = _MASK[:, -12:] = False


def _measure(path: Path, info: dict) -> np.ndarray:
    """Zoom of every frame relative to the clip's first frame, smoothed."""
    lows = [_low(f) for f in _frames(path, info["w"], info["h"])]
    ref = lows[0]
    candidates = [_zoom(ref, s) for s in SCALES]
    raw = np.array([SCALES[int(np.argmin([np.mean(np.abs(c[_MASK] - f[_MASK])) for c in candidates]))] for f in lows])
    # Smooth drift with s(0) = 1: s - 1 = a*t + b*t^2 (least squares).
    t = np.arange(len(raw)) / max(1, len(raw) - 1)
    a, b = np.linalg.lstsq(np.stack([t, t * t], 1), raw - 1.0, rcond=None)[0]
    return 1.0 + a * t + b * t * t


def _states(action: str) -> tuple[str, str]:
    """(start state, end state): "base" or a prop scene name."""
    if action.startswith("scene_") and action.endswith("_in"):
        scene = action[len("scene_"):-len("_in")]
        return actions.SCENE_PARENT.get(scene, "base"), scene
    if action.startswith("scene_") and action.endswith("_out"):
        scene = action[len("scene_"):-len("_out")]
        return scene, actions.SCENE_PARENT.get(scene, "base")
    scene = actions.BY_KEY[action].scene if action in actions.BY_KEY else ""
    return (scene or "base"), (scene or "base")


def _first_frame(path: Path) -> np.ndarray:
    info = _probe(path)
    return next(_frames(path, info["w"], info["h"])).copy()


def stabilize(outfit: str) -> None:
    root = MEDIA_DIR / "outfits" / outfit
    orig = root / ".orig"
    clips: list[tuple[str, Path, Path]] = []  # (action, working file, original)
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in sorted(folder.glob("*.mp4")):
            if f.stem.endswith("_rev") or f.stem.endswith("_loop"):
                continue
            src = orig / folder.name / f.name
            if not src.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, src)
            clips.append((folder.name, f, src))

    # Canonical frame per state, taken from the originals.
    # "base" is the idle frame; a prop scene's frame comes from one of its action clips.
    canon: dict[str, np.ndarray] = {}
    idle = [src for a, _, src in clips if a == "idle"]
    canon["base"] = _first_frame(idle[0] if idle else next(src for a, _, src in clips if _states(a)[0] == "base"))
    for action, _, src in clips:
        start, _end = _states(action)
        if start != "base" and not action.startswith("scene_") and start not in canon:
            canon[start] = _first_frame(src)

    print("measuring zoom drift…", flush=True)
    drift = {}
    for action, _, src in clips:
        drift[src] = _measure(src, _probe(src))
        print(f"  {action:18s} end zoom {drift[src][-1]:.4f}", flush=True)
    target = max(1.0, max(float(d.max()) for d in drift.values())) + 0.002
    print(f"fixed framing: x{target:.4f}", flush=True)
    canon_z = {k: _zoom(v, target) for k, v in canon.items()}

    for action, work, src in clips:
        info = _probe(src)
        s = drift[src]
        n = len(s)
        start, end = _states(action)
        tmp = work.with_suffix(".tmp.mp4")
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info['w']}x{info['h']}",
               "-r", f"{info['fps']:.6f}", "-i", "-"]
        if info["audio"]:
            cmd += ["-i", str(src), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i, frame in enumerate(_frames(src, info["w"], info["h"])):
            g = _zoom(frame, target / s[min(i, n - 1)]).astype(np.float32)
            if start in canon_z and i < BLEND:
                w = i / BLEND
                g = canon_z[start] * (1 - w) + g * w
            if end in canon_z and i >= n - BLEND:
                w = (i - (n - BLEND) + 1) / BLEND
                g = g * (1 - w) + canon_z[end] * w
            enc.stdin.write(np.clip(g, 0, 255).astype(np.uint8).tobytes())
        enc.stdin.close()
        enc.wait()
        tmp.replace(work)
        print(f"  stabilized {action}/{work.name}", flush=True)
        rev = work.with_name(f"{work.stem}_rev.mp4")
        if rev.exists():
            make_reverse(work)
            print(f"  rebuilt {rev.name}", flush=True)
    for loop in root.glob("*/*_loop.mp4"):
        loop.unlink()  # stale joined loops; the stage now plays an idle playlist instead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outfit", required=True, help="folder under asmr/media/outfits/")
    stabilize(parser.parse_args().outfit)


if __name__ == "__main__":
    main()
