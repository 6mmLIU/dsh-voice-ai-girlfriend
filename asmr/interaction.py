"""Viewer interaction events from live platforms -> what the host does about them.

Every platform adapter (asmr/danmaku/*) posts normalized events to the stage:

    POST /api/event {"kind": "comment" | "gift" | "join" | "follow" | "like",
                     "user": "...", "text": "...", "gift": "...", "count": 1,
                     "id": "<platform message id, optional>", "platform": "wechat_channels"}

This module decides, per event, whether to answer and how:
  * comment - filtered (blocklist, length, duplicates, per-viewer cooldown) and
              handed to the LLM director for a whispered reply
  * gift    - a quick thank-you line (template, no LLM wait); repeats of the
              same gift from the same viewer within a few seconds are merged
  * follow  - a thank-you line, rate limited
  * join    - an occasional welcome, only when nothing else is waiting
  * like    - counted; now and then a "thanks for the likes" when idle
Settings: asmr-config.json -> "interaction" (see DEFAULTS).
"""
from __future__ import annotations

import random
import time
from collections import OrderedDict
from dataclasses import dataclass

from .config import CONFIG

DEFAULTS = {
    "user_cooldown_sec": 20,  # one answered comment per viewer per this many seconds
    "welcome_every_sec": 45,  # at most one welcome line this often
    "follow_every_sec": 10,
    "likes_every_sec": 180,
    "gift_merge_sec": 4,  # same viewer + same gift within this window = one thank-you
    "max_comment_chars": 80,
    "blocklist": [],  # comments containing any of these are ignored
    "ignore_users": [],  # e.g. the host's own account
    "templates": {
        "gift": [
            "[左]谢谢{user}送的{gift}…[停顿][右]好开心呀…",
            "[右]{user}…谢谢你的{gift}[中]我会好好陪着你的…",
            "[中]哇…{user}送了{gift}…谢谢你哦…",
        ],
        "follow": [
            "[左]谢谢{user}的关注…[右]以后也一起放松吧…",
            "[中]{user}…欢迎你留下来…",
        ],
        "join": [
            "[左]欢迎{user}…[右]慢慢放松就好…",
            "[右]{user}来啦…[中]找个舒服的姿势吧…",
            "[中]晚上好呀{user}…",
        ],
        "likes": [
            "[中]谢谢大家的点赞…[停顿]我们继续放松吧…",
        ],
    },
}


def _cfg() -> dict:
    merged = dict(DEFAULTS)
    merged.update(CONFIG.get("interaction", {}))
    merged["templates"] = {**DEFAULTS["templates"], **CONFIG.get("interaction", {}).get("templates", {})}
    return merged


@dataclass
class Decision:
    action: str  # "reply" (send to the director) | "script" (play this tagged script) | "drop"
    script: str = ""
    reason: str = ""
    front: bool = False


class Interaction:
    def __init__(self) -> None:
        self.seen: OrderedDict[str, float] = OrderedDict()  # message ids / content keys -> time
        self.last_reply: dict[str, float] = {}
        self.last_gift: dict[tuple[str, str], float] = {}
        self.last_welcome = 0.0
        self.last_follow = 0.0
        self.last_likes = 0.0
        self.likes = 0

    def _dupe(self, key: str, window: float) -> bool:
        now = time.monotonic()
        while self.seen and next(iter(self.seen.values())) < now - 600:
            self.seen.popitem(last=False)
        if key in self.seen and now - self.seen[key] < window:
            return True
        self.seen[key] = now
        self.seen.move_to_end(key)
        return False

    def decide(self, ev: dict, busy: bool) -> Decision:
        """`busy`: replies or scripts are already waiting (welcomes / like-thanks then wait)."""
        cfg = _cfg()
        kind = ev.get("kind", "comment")
        user = (ev.get("user") or "朋友").strip()[:16]
        now = time.monotonic()
        if user in cfg["ignore_users"]:
            return Decision("drop", reason="ignored user")
        if ev.get("id") and self._dupe(f"id:{ev['id']}", 600):
            return Decision("drop", reason="duplicate id")

        if kind == "comment":
            text = (ev.get("text") or "").strip()
            if not text or len(text) > cfg["max_comment_chars"]:
                return Decision("drop", reason="empty or too long")
            if any(w and w in text for w in cfg["blocklist"]):
                return Decision("drop", reason="blocklist")
            if self._dupe(f"c:{user}:{text}", 60):
                return Decision("drop", reason="duplicate comment")
            if now - self.last_reply.get(user, -1e9) < cfg["user_cooldown_sec"]:
                return Decision("drop", reason="viewer cooldown")
            self.last_reply[user] = now
            return Decision("reply")

        tpl = cfg["templates"]
        if kind == "gift":
            gift = (ev.get("gift") or "礼物").strip()[:12]
            key = (user, gift)
            if now - self.last_gift.get(key, -1e9) < cfg["gift_merge_sec"]:
                self.last_gift[key] = now
                return Decision("drop", reason="merged gift")
            self.last_gift[key] = now
            return Decision("script", random.choice(tpl["gift"]).format(user=user, gift=gift), front=True)
        if kind == "follow":
            if now - self.last_follow < cfg["follow_every_sec"]:
                return Decision("drop", reason="follow rate limit")
            self.last_follow = now
            return Decision("script", random.choice(tpl["follow"]).format(user=user))
        if kind == "join":
            if busy or now - self.last_welcome < cfg["welcome_every_sec"]:
                return Decision("drop", reason="welcome rate limit")
            self.last_welcome = now
            return Decision("script", random.choice(tpl["join"]).format(user=user))
        if kind == "like":
            self.likes += int(ev.get("count") or 1)
            if busy or self.likes < 10 or now - self.last_likes < cfg["likes_every_sec"]:
                return Decision("drop", reason="likes counted")
            self.last_likes, self.likes = now, 0
            return Decision("script", random.choice(tpl["likes"]))
        return Decision("drop", reason=f"unknown kind {kind}")
