"""Director: turns viewer danmaku into a tagged whisper script, and parses it.

Script format (what the LLM writes, and what the operator can type by hand):

    [左]晚安呀，小明…[耳朵按摩][右]今天辛苦了[停顿][敲木盒20]

Position tags set where the voice / sound sits around the listener's head:
    [左] left ear, close   [右] right ear, close
    [中] in front, close   [绕] sweeps from left ear to right ear
Action tags (see actions.py for the full catalog) play a long, sustained
trigger with its sound and stage clip. A number after the tag sets its
length in seconds: [耳朵按摩30]. Without one, the action's configured
duration is used.
Light tags (visual only, they play over what follows):
    [灯光] alternating left/right light pulses
    [引导] a slow breathing light to follow: brightens on the inhale, fades
           on the exhale (pair it with "吸气…呼气…")
    [停顿] a silent pause
"""
from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field

import httpx

from . import actions
from .config import CONFIG

logger = logging.getLogger("asmr.director")

POSITIONS = {"左": "L", "右": "R", "中": "C", "绕": "SWEEP"}
VISUALS = {"灯光": "lights", "引导": "guide"}
PAUSE = "停顿"
MAX_SCRIPT_CHARS = 200

_TAG_RE = re.compile(r"(\[[^\[\]]{1,8}\])")
_SECS_RE = re.compile(r"^(.*?)(\d{1,3})$")


@dataclass
class Segment:
    kind: str  # "speech" | "sound" | "pause"
    pos: str = "C"  # L / R / C / SWEEP
    text: str = ""
    name: str = ""  # action key (or "lights"/"guide") for kind == "sound"
    seconds: float | None = None  # explicit length from the tag, e.g. [耳朵按摩30]


def parse_script(script: str) -> list[Segment]:
    """Split a tagged script into speech / sound / pause segments.

    Unknown tags are dropped; text outside tags becomes speech at the
    current position. The position persists until the next position tag.
    """
    pos = "C"
    out: list[Segment] = []
    for token in _TAG_RE.split(script[:MAX_SCRIPT_CHARS]):
        if not token:
            continue
        if token.startswith("[") and token.endswith("]"):
            tag = token[1:-1].strip()
            secs = None
            m = _SECS_RE.match(tag)
            if m and m.group(1):
                tag, secs = m.group(1), float(m.group(2))
            if tag in POSITIONS:
                pos = POSITIONS[tag]
            elif tag in actions.BY_TAG:
                out.append(Segment(kind="sound", pos=pos, name=actions.BY_TAG[tag].key, seconds=secs))
            elif tag in VISUALS:
                out.append(Segment(kind="sound", pos=pos, name=VISUALS[tag], seconds=secs))
            elif tag == PAUSE:
                out.append(Segment(kind="pause", pos=pos))
            continue
        text = token.strip()
        if any(ch.isalnum() for ch in text):
            out.append(Segment(kind="speech", pos=pos, text=text))
    return out


def strip_tags(script: str) -> str:
    """Display text: the script without any tags."""
    return _TAG_RE.sub("", script).strip()


def _action_menu() -> str:
    groups: dict[str, list[str]] = {}
    for a in actions.CATALOG:
        groups.setdefault(a.group, []).append(f"[{a.tag}] {a.describe}")
    return "\n".join(f"   {g}：" + "；".join(items) for g, items in groups.items())


def _system_prompt() -> str:
    persona = CONFIG.get("persona", {})
    name = persona.get("name", "小雅")
    style = persona.get("style", "温柔、安静的助眠陪伴主播")
    return f"""你是 ASMR 直播间的主播「{name}」，{style}。
你在用耳语回应直播间观众的弹幕。观众戴着耳机，你的声音会出现在他们的左耳、右耳或正前方。
你面前有一个双耳麦克风（两只硅胶假耳朵，就是观众的耳朵），桌上有各种 ASMR 道具。

输出规则（必须遵守）：
1. 只输出一行要被念出来的台词，夹带标签，不要任何解释、引号或换行。
2. 台词全部是很轻的耳语口吻：短句，每句不超过 15 个字，多用"…"放慢节奏。整段不超过 50 个字。
3. 开头先轻声叫出观众的昵称。
4. 用位置标签安排声音方位：[左] 左耳 [右] 右耳 [中] 正前方 [绕] 从左耳绕到右耳。至少换一次方位。
5. 在句子之间插入 1-2 个动作标签。动作会慢慢地持续做很久（十几到几十秒），所以一段回复里动作不要超过 2 个。
   动作前面的位置标签决定在哪只耳朵 / 哪一侧做，例如 [左][耳朵按摩] 就是按摩左耳。
   观众点名想听某种声音时，就做那个动作；可以在标签后写秒数指定时长，例如 [敲木盒30]。
   可用动作：
{_action_menu()}
6. 灯光标签：[灯光] 左右交替的灯光；[引导] 跟着呼吸灯慢慢吸气呼气（观众说睡不着、焦虑时用，配合"吸气…呼气…"）。[停顿] 安静停一下。
7. 不要使用除上面以外的任何标签。

示例：
[左]小明…晚上好呀[耳朵按摩][右]今天也辛苦了吧…[停顿][中]想听木头的声音吗…[敲木盒]
[中]睡不着吗…[引导]跟着灯光…慢慢吸气…[停顿]再慢慢呼气…[停顿][左][棉签]"""


_FALLBACK = [
    "[左]{user}…你来啦[耳朵按摩][右]欢迎欢迎…[停顿][中]慢慢放松哦",
    "[右]{user}…收到你的话啦[刷耳朵][左]我在认真听哦…",
    "[绕]{user}…谢谢你陪着我[灯光][中]今晚也要好好休息[敲木盒]",
    "[中]{user}…我们一起放松一下[引导]慢慢吸气…[停顿]慢慢呼气…[停顿][左][棉签]",
    "[中]{user}…给你翻翻书吧[翻书][右]慢慢的…不着急",
]


@dataclass
class Director:
    history: list[dict] = field(default_factory=list)

    def _api_key(self) -> str:
        llm = CONFIG.get("llm", {})
        return llm.get("api_key") or os.environ.get(llm.get("api_key_env", "DEEPSEEK_API_KEY"), "")

    @property
    def has_llm(self) -> bool:
        return bool(self._api_key())

    async def reply(self, user: str, text: str) -> str:
        """Return a tagged whisper script answering one danmaku."""
        user = (user or "朋友").strip()[:12]
        text = (text or "").strip()[:100]
        if not self.has_llm:
            return random.choice(_FALLBACK).format(user=user)
        llm = CONFIG.get("llm", {})
        message = {"role": "user", "content": f"观众「{user}」发来弹幕：{text}"}
        messages = [{"role": "system", "content": _system_prompt()}, *self.history, message]
        try:
            async with httpx.AsyncClient(timeout=llm.get("timeout_sec", 30)) as client:
                resp = await client.post(
                    llm.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key()}"},
                    json={
                        "model": llm.get("model", "deepseek-chat"),
                        "messages": messages,
                        "temperature": llm.get("temperature", 0.8),
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                script = resp.json()["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001 - fall back so the stream never goes silent
            logger.exception("LLM call failed, using a fallback line")
            return random.choice(_FALLBACK).format(user=user)
        script = " ".join(script.split())[:MAX_SCRIPT_CHARS]
        if not any(s.kind == "speech" for s in parse_script(script)):
            logger.warning("LLM script had no speech, using fallback: %r", script)
            return random.choice(_FALLBACK).format(user=user)
        keep = int(llm.get("max_history", 8)) * 2
        self.history = [*self.history, message, {"role": "assistant", "content": script}][-keep:]
        return script


def idle_script() -> str:
    """Something to play when the chat is quiet: mostly sounds, sometimes a line."""
    idle = CONFIG.get("idle", {})
    lines = idle.get("lines") or []
    if lines and random.random() < float(idle.get("whisper_chance", 0.35)):
        return random.choice(lines)
    # One long trigger, sometimes with the lights; cycles through the catalog.
    action = random.choice([a for a in actions.CATALOG if a.key != "breath"])
    side = random.choice(["[左]", "[右]", "[中]"]) if action.placement == "ear" else "[中]"
    light = random.choice(["", "", "[灯光]", "[引导]"])
    return f"{light}{side}[{action.tag}]"
