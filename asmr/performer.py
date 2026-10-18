"""The live performer: strings playback units together, one at a time.

Nothing longer than one unit (a take of an action, a whispered line, a
transition) is ever committed in advance, so the operator can:
  * start / switch to any action at any time  ("想换就换")
  * stop the current action at any time        ("想停就停")
  * give an action a length, or let it run until stopped, and change that
    length while it runs
A stop or switch takes effect at the end of the take in progress (a few
seconds at most), then the current scene is closed properly (tool put down,
props cleared) before anything else starts, so the picture never jumps.

Scripts from the director (danmaku replies) and from the operator become a
queue of steps. A whispered line during an ongoing action rides over that
action's takes; an action step replaces the current action.

The next unit is sent to the stage shortly before the current one ends
(LOOKAHEAD), so playback is gapless. Units already sent but not yet started
are recalled ("flush") when a command arrives, and the state is rolled back.
"""
from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from . import actions, media, render
from .config import CONFIG
from .director import parse_script, strip_tags

logger = logging.getLogger("asmr.performer")

LOOKAHEAD = 1.5  # seconds before the current unit ends that the next one is sent
_script_ids = itertools.count(1)


@dataclass
class Step:
    kind: str  # speech | action | pause | visual
    pos: str = "C"
    text: str = ""
    key: str = ""
    seconds: float | None = None
    speech: asyncio.Task | None = None
    meta: dict = field(default_factory=dict)
    script: int = 0  # which script the step came from


@dataclass
class Sent:
    unit: dict
    start: float
    end: float
    started: bool
    snapshot: dict
    consumed: list[Step]


class Performer:
    def __init__(self, broadcast, log) -> None:
        self.broadcast = broadcast
        self.log = log
        self.steps: deque[Step] = deque()
        # key, pos, length (None = until stopped), started (first take; None before), takes, last_url.
        # The clock starts with the first take, so set-down transitions don't eat into the length.
        self.active: dict | None = None
        self.open: list[str] = []  # scenes currently open, outermost first
        self.visual: list[dict] = []  # light cues waiting for the next unit
        self.whisper_last: str | None = None
        self.timeline_end = time.monotonic()
        self.sent: list[Sent] = []
        self.wake = asyncio.Event()
        self.last_sound = time.monotonic()

    # ── input ──────────────────────────────────────────────────────────────
    def enqueue_script(self, script: str, meta: dict | None = None, front: bool = False) -> None:
        """Turn a tagged script into steps (speech starts rendering right away)."""
        steps = []
        sid = next(_script_ids)
        for seg in parse_script(script):
            if seg.kind == "speech":
                task = asyncio.ensure_future(render.speech_audio(seg.text, seg.pos))
                steps.append(Step("speech", seg.pos, text=seg.text, speech=task, meta=dict(meta or {}), script=sid))
            elif seg.kind == "pause":
                steps.append(Step("pause", seg.pos, script=sid))
            elif seg.name in ("lights", "guide"):
                steps.append(Step("visual", seg.pos, key=seg.name, seconds=seg.seconds, script=sid))
            else:
                steps.append(Step("action", seg.pos, key=seg.name, seconds=seg.seconds, script=sid))
        if front:
            self.steps.extendleft(reversed(steps))
        else:
            self.steps.extend(steps)
        self.wake.set()

    def perform(self, key: str, pos: str = "C", seconds: float | None = None, forever: bool = False) -> None:
        """Start or switch to an action now (after the take in progress)."""
        self._flush()
        length = None if forever else (seconds if seconds else actions.seconds(key))
        self.active = {"key": key, "pos": pos, "length": length, "started": None, "takes": 0, "last_url": None,
                       "script": 0}
        self.wake.set()

    def stop(self) -> None:
        """Stop the current action (after the take in progress) and clear its scene."""
        self._flush()
        self.active = None
        self.wake.set()

    def set_length(self, seconds: float | None) -> None:
        """Change how long the current action runs in total (None = until stopped)."""
        if self.active:
            self.active["length"] = None if seconds is None else max(2.0, seconds)
            self.wake.set()

    def status(self) -> dict:
        now = time.monotonic()
        a = self.active
        playing = next((s.unit for s in reversed(self.sent) if s.start <= now < s.end), None)
        elapsed = max(0.0, now - a["started"]) if a and a["started"] is not None else 0.0
        return {
            "action": a and {
                "key": a["key"], "tag": actions.BY_KEY[a["key"]].tag if a["key"] in actions.BY_KEY else a["key"],
                "pos": a["pos"], "elapsed": round(elapsed, 1),
                "remaining": None if a["length"] is None else round(max(0.0, a["length"] - elapsed), 1),
                "forever": a["length"] is None,
            },
            "scenes": list(self.open),
            "queued": len(self.steps),
            "playing": playing and {"id": playing["id"], "what": playing.get("what", "")},
        }

    def on_started(self, unit_id: str) -> None:
        """The stage reports a unit's real start; re-sync the timeline if it lagged."""
        now = time.monotonic()
        for i, s in enumerate(self.sent):
            if s.unit["id"] == unit_id and not s.started:
                s.started = True
                lag = now - s.start
                if abs(lag) > 0.15:
                    for later in self.sent[i:]:
                        later.start += lag
                        later.end += lag
                    self.timeline_end += lag
                return

    # ── output ─────────────────────────────────────────────────────────────
    def _snapshot(self) -> dict:
        return {"active": copy.deepcopy(self.active), "open": list(self.open), "visual": list(self.visual),
                "whisper_last": self.whisper_last}

    def _flush(self) -> None:
        """Recall units sent to the stage that have not started yet."""
        now = time.monotonic()
        pending = [s for s in self.sent if not s.started and s.start > now + 0.05]
        if not pending:
            return
        first = pending[0]
        snap = first.snapshot
        self.active, self.open, self.visual, self.whisper_last = snap["active"], snap["open"], snap["visual"], snap["whisper_last"]
        for s in reversed(pending):
            self.steps.extendleft(reversed(s.consumed))
        self.sent = [s for s in self.sent if s not in pending]
        self.timeline_end = first.start
        asyncio.ensure_future(self.broadcast({"type": "flush", "ids": [s.unit["id"] for s in pending]}))

    async def run(self) -> None:
        idle_after = float(CONFIG.get("idle", {}).get("after_sec", 12))
        while True:
            self.wake.clear()
            now = time.monotonic()
            if self.timeline_end - now > LOOKAHEAD:
                await self._sleep(min(self.timeline_end - now - LOOKAHEAD, 1.0))
                continue
            t0 = max(now, self.timeline_end)
            snap = self._snapshot()
            consumed: list[Step] = []
            try:
                unit = await self._next_unit(t0, consumed)
            except Exception:  # noqa: BLE001 - never let the show stop
                logger.exception("unit failed")
                unit = None
            if unit is None:
                if consumed:  # steps were used up without producing a unit (e.g. an action that starts next round)
                    continue
                if now - self.timeline_end > idle_after and not self.steps and self.active is None and not self.open:
                    from .director import idle_script
                    self.enqueue_script(idle_script(), {"source": "idle"})
                    await self.log("空闲节目")
                    continue
                await self._sleep(0.3)
                continue
            t0 = max(time.monotonic(), self.timeline_end)
            if self.visual:
                unit["cues"] = [*self.visual, *unit["cues"]]
                self.visual = []
            self.sent.append(Sent(unit, t0, t0 + unit["duration"], False, snap, consumed))
            self.sent = [s for s in self.sent if s.end > time.monotonic() - 30]
            self.timeline_end = t0 + unit["duration"]
            await self.broadcast(unit)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.wake.wait(), timeout=max(0.05, seconds))
        except asyncio.TimeoutError:
            pass

    # ── deciding what comes next ───────────────────────────────────────────
    def _target_path(self) -> list[str]:
        a = self.active
        if not a or not media.files(a["key"]):
            return list(self.open) if a else []  # sound-only action: stay where we are
        scene = actions.BY_KEY[a["key"]].scene if a["key"] in actions.BY_KEY else ""
        return actions.scene_path(scene)

    def _transition_toward(self, target: list[str]) -> dict | None:
        """Close or open one scene on the way from the open scenes to `target`."""
        common = 0
        while common < min(len(self.open), len(target)) and self.open[common] == target[common]:
            common += 1
        if len(self.open) > common:
            scene = self.open.pop()
            return render.unit_transition(f"scene_{scene}_out", {"what": f"收起 {scene}"}) or {"skip": True}
        if len(target) > common:
            scene = target[common]
            self.open.append(scene)
            return render.unit_transition(f"scene_{scene}_in", {"what": f"拿出 {scene}"}) or {"skip": True}
        return None

    async def _next_unit(self, t0: float, consumed: list[Step]) -> dict | None:
        for _ in range(8):  # skip over steps / missing transitions until a unit comes out
            unit = await self._decide(t0, consumed)
            if unit is None or not unit.get("skip"):
                return unit
        return None

    async def _decide(self, t0: float, consumed: list[Step]) -> dict | None:
        # Light cues ride on whatever unit comes next.
        while self.steps and self.steps[0].kind == "visual":
            v = self.steps.popleft()
            consumed.append(v)
            dur = v.seconds or (20.0 if v.key == "guide" else 8.0)
            self.visual.append({"t": 0.0, "dur": dur, "kind": "visual", "name": v.key})

        a = self.active
        if a and a["length"] is not None and a["started"] is not None and t0 >= a["started"] + a["length"] - 0.5:
            await self.log(f"动作结束：{a['key']}")
            self.active = a = None

        step = self.steps[0] if self.steps else None
        if step and step.kind == "action" and a and a.get("script") == step.script:
            # Actions in one script run one after another: this one waits for
            # the current one to finish (a new script or the operator switches at once).
            step = None
        if step and step.kind == "action":
            self.steps.popleft()
            consumed.append(step)
            length = step.seconds or actions.seconds(step.key)
            self.active = a = {"key": step.key, "pos": step.pos, "length": length, "started": None,
                               "takes": 0, "last_url": None, "script": step.script}
            await self.log(f"开始动作：{step.key} {length:.0f}s")
            return {"skip": True}

        if step and step.kind == "pause":
            if a:  # a pause during an action: she just keeps doing it
                self.steps.popleft()
                consumed.append(step)
                return {"skip": True}
            if self.open:
                return self._transition_toward([])
            self.steps.popleft()
            consumed.append(step)
            return render.unit_pause(1.0, {"what": "停顿"})

        if step and step.kind == "speech":
            if not a and self.open:  # talk from the idle pose: clear the desk first
                return self._transition_toward([])
            if a and self.open != self._target_path():  # get the action's scene ready first
                return self._transition_toward(self._target_path())
            if not step.speech.done():
                if a:  # keep the action going while the line is being voiced
                    return self._action_take(a, t0)
                await asyncio.wait({step.speech}, timeout=0.3)
                if not step.speech.done():
                    return None
            self.steps.popleft()
            consumed.append(step)
            speech = step.speech.result()
            meta = {"what": f"说：{step.text}", "display": step.text, **step.meta}
            if a and a["started"] is None:
                a["started"] = t0
            if a and media.files(a["key"]) and self.open == self._target_path():
                unit, a["last_url"] = render.unit_speech(speech, step.text, step.pos, meta, a)
                a["takes"] += 1
            elif a:
                unit, _ = render.unit_speech(speech, step.text, step.pos, meta, a)
            else:
                unit, _ = render.unit_speech(speech, step.text, step.pos, meta)
            self.last_sound = time.monotonic()
            return unit

        if a:
            target = self._target_path()
            if self.open != target:
                return self._transition_toward(target)
            return self._action_take(a, t0)

        if self.open:
            return self._transition_toward([])
        return None

    def _action_take(self, a: dict, t0: float) -> dict:
        if a["started"] is None:
            a["started"] = t0
        unit, a["last_url"] = render.unit_take(a["key"], a["pos"], a["takes"], a["last_url"],
                                               {"what": f"{a['key']} 第 {a['takes'] + 1} 下"})
        a["takes"] += 1
        self.last_sound = time.monotonic()
        return unit


def display_text(script: str) -> str:
    return strip_tags(script)
