"""The ASMR action catalog: every trigger the host can perform.

Each action has a script tag, a default duration (long on purpose: an ASMR
trigger is a slow, sustained activity, not a one-off sound), and a placement
that decides where the sound moves around the listener's head:

  ear        right against one dummy ear; alternates left/right ear in turns
  alternate  discrete hits that alternate left and right hands
  sweep      a slow movement that drifts from one ear to the other and back
  center     in front of the mic, close

Actions that need props on the desk belong to a prop scene (scene = "knock",
"fluffy", "soft"...). Their clips start and end on that scene's frame, so the
renderer plays the scene's transition clips around them: scene_<name>_in
(she sets the props down) before, scene_<name>_out (she clears them) after.

Durations can be changed three ways (most specific wins):
  1. in the script: a number after the tag, e.g. [耳朵按摩30] = 30 seconds
  2. live from the control panel (saved to asmr/out/actions.json)
  3. asmr-config.json -> "actions": {"ear_massage": 25, ...}
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import CONFIG, OUT_DIR


@dataclass(frozen=True)
class Action:
    key: str
    tag: str
    group: str
    seconds: float
    placement: str
    describe: str  # for the LLM director and the clip prompts
    scene: str = ""  # prop scene whose clips start/end with the props on the desk


CATALOG: list[Action] = [
    # ① 硅胶耳朵 + 双耳麦克风
    Action("ear_massage", "耳朵按摩", "耳朵", 24, "ear", "手指揉捏硅胶耳廓、指腹轻搓耳垂"),
    Action("ear_pick", "掏耳朵", "耳朵", 24, "ear", "指尖在耳道口轻轻掏、刮"),
    Action("ear_swab", "棉签", "耳朵", 20, "ear", "棉签轻扫硅胶耳朵表面和耳道口"),
    Action("ear_brush", "刷耳朵", "耳朵", 20, "ear", "软刷刷硅胶耳廓"),
    # ② 化妆刷、羽毛刷
    Action("brush", "刷子", "刷子", 18, "sweep", "化妆刷轻扫麦克风"),
    Action("feather", "羽毛刷", "刷子", 18, "sweep", "羽毛刷在麦克风和桌面上缓慢扫动"),
    Action("face_brush", "刷脸", "刷子", 15, "center", "刷子靠近镜头，像在轻刷观众的脸"),
    # ③ 木块、玻璃瓶、化妆品容器
    Action("tap", "敲击", "敲击", 14, "alternate", "指尖左右手交替轻敲麦克风和桌面"),
    Action("wood", "敲木盒", "敲击", 14, "alternate", "指尖或美甲缓慢敲击木盒，空心木声", "knock"),
    Action("glass", "敲玻璃", "敲击", 14, "alternate", "美甲轻敲、轻刮玻璃瓶", "knock"),
    Action("cap", "拧瓶盖", "敲击", 12, "center", "慢慢旋转化妆品瓶盖", "knock"),
    # ④ 毛绒麦套、海绵、丝绒布
    Action("fluffy", "毛绒", "柔软", 20, "ear", "手掌抚摸毛绒麦套、指尖轻抓绒毛", "fluffy"),
    Action("sponge", "海绵", "柔软", 15, "center", "海绵轻压、回弹", "soft"),
    Action("velvet", "丝绒", "柔软", 18, "sweep", "丝绒布在麦克风上缓慢摩擦", "soft"),
    Action("felt", "毛毡", "柔软", 18, "sweep", "双手搓毛毡垫"),
    # ⑤ 纸张、包装袋、气泡膜
    Action("page", "翻书", "纸张", 18, "center", "缓慢翻书页", "paper"),
    Action("paper", "揉纸", "纸张", 14, "sweep", "慢慢揉捏包装纸", "paper"),
    Action("bubble", "气泡膜", "纸张", 12, "alternate", "一颗一颗按压气泡膜", "paper"),
    Action("tape", "胶带", "纸张", 10, "sweep", "慢慢撕开胶带", "paper"),
    # ⑥ 手套、护手霜、手部动作
    Action("gloves", "手套", "手部", 18, "alternate", "戴着乳胶手套搓手、指尖摩擦", "gloves"),
    Action("hands", "搓手", "手部", 14, "center", "双手缓慢搓手，在镜头前轻柔移动", "lotion"),
    Action("lotion", "护手霜", "手部", 16, "center", "挤护手霜，双手慢慢揉开", "lotion"),
    # 其他
    Action("breath", "呼吸", "其他", 6, "center", "一次缓慢的呼吸"),
]

BY_KEY = {a.key: a for a in CATALOG}
BY_TAG = {a.tag: a for a in CATALOG}

_STATE = OUT_DIR / "actions.json"


def _overrides() -> dict:
    try:
        return json.loads(_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def seconds(key: str) -> float:
    """Current duration for an action: control-panel override > config > default."""
    live = _overrides()
    scale = float(live.get("_scale", 1.0))
    if key in live:
        return float(live[key])
    configured = CONFIG.get("actions", {}).get(key)
    base = float(configured) if configured is not None else BY_KEY[key].seconds
    return round(base * scale, 1)


def set_seconds(key: str | None, value: float | None, scale: float | None = None) -> None:
    live = _overrides()
    if scale is not None:
        live["_scale"] = max(0.25, min(4.0, float(scale)))
    if key in BY_KEY:
        if value is None:
            live.pop(key, None)
        else:
            live[key] = max(2.0, min(120.0, float(value)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _STATE.write_text(json.dumps(live, ensure_ascii=False, indent=1), encoding="utf-8")


def listing() -> dict:
    live = _overrides()
    return {
        "scale": float(live.get("_scale", 1.0)),
        "actions": [
            {"key": a.key, "tag": a.tag, "group": a.group, "seconds": seconds(a.key),
             "default": a.seconds, "pinned": a.key in live}
            for a in CATALOG
        ],
    }
