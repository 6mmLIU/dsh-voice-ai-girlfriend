"""Director: turns viewer danmaku into a tagged whisper script, and parses it.

Script format (what the LLM writes, and what the operator can type by hand):

    [左]晚安呀，小明…[敲击][右]今天辛苦了[停顿][毛毡]

Position tags set where the voice / sound sits around the listener's head:
    [左] left ear, close   [右] right ear, close
    [中] in front, close   [绕] sweeps from left ear to right ear
Action tags insert a trigger sound and the matching stage animation:
    [敲击] alternating fingertip taps   [毛毡] felt rubbing
    [刷子] brush strokes                [呼吸] a soft breath
    [灯光] alternating left/right light pulses (visual only)
    [停顿] a silent pause
"""
from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field

import httpx

from .config import CONFIG

logger = logging.getLogger("asmr.director")

POSITIONS = {"左": "L", "右": "R", "中": "C", "绕": "SWEEP"}
SOUNDS = {"敲击": "tap", "毛毡": "felt", "刷子": "brush", "呼吸": "breath", "灯光": "lights"}
PAUSE = "停顿"
MAX_SCRIPT_CHARS = 160

_TAG_RE = re.compile(r"(\[[^\[\]]{1,4}\])")


@dataclass
class Segment:
    kind: str  # "speech" | "sound" | "pause"
    pos: str = "C"  # L / R / C / SWEEP
    text: str = ""
    name: str = ""  # sound name for kind == "sound"


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
            if tag in POSITIONS:
                pos = POSITIONS[tag]
            elif tag in SOUNDS:
                out.append(Segment(kind="sound", pos=pos, name=SOUNDS[tag]))
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


def _system_prompt() -> str:
    persona = CONFIG.get("persona", {})
    name = persona.get("name", "小雅")
    style = persona.get("style", "温柔、安静的助眠陪伴主播")
    return f"""你是 ASMR 直播间的主播「{name}」，{style}。
你在用耳语回应直播间观众的弹幕。观众戴着耳机，你的声音会出现在他们的左耳、右耳或正前方。

输出规则（必须遵守）：
1. 只输出一行要被念出来的台词，夹带标签，不要任何解释、引号或换行。
2. 台词全部是很轻的耳语口吻：短句，每句不超过 15 个字，多用"…"放慢节奏。整段不超过 50 个字。
3. 开头先轻声叫出观众的昵称。
4. 用位置标签安排声音方位：[左] 左耳 [右] 右耳 [中] 正前方 [绕] 从左耳绕到右耳。至少换一次方位。
5. 在句子之间插入 1-2 个动作标签：[敲击] 指尖轻敲 [毛毡] 摩擦毛毡 [刷子] 刷子轻扫 [呼吸] 轻呼吸 [灯光] 左右交替的灯光 [停顿] 安静停一下。
6. 不要使用除上面以外的任何标签。

示例：
[左]小明…晚上好呀[敲击][右]今天也辛苦了吧…[停顿][中]慢慢放松就好[毛毡]"""


_FALLBACK = [
    "[左]{user}…你来啦[敲击][右]欢迎欢迎…[停顿][中]慢慢放松哦[毛毡]",
    "[右]{user}…收到你的话啦[刷子][左]我在认真听哦…[呼吸]",
    "[绕]{user}…谢谢你陪着我[灯光][中]今晚也要好好休息[毛毡]",
    "[左]嗯…{user}说得对[敲击][右]那我们再安静一会儿…[停顿]",
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
    return random.choice([
        "[中][毛毡][停顿][毛毡]",
        "[中][敲击][停顿][敲击]",
        "[绕][刷子][停顿][灯光]",
        "[左][毛毡][右][毛毡]",
        "[中][灯光][呼吸][灯光]",
    ])
