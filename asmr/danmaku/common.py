"""Shared pieces for live-platform adapters: event extraction and the stage sink.

Platforms change their payloads often, so extraction is heuristic: walk any
JSON value and pick out objects that look like a viewer message - a nickname
plus a comment text, or a nickname plus a gift. Adapters can add precise
parsers on top once the real payload is known (see `--dump` in each adapter).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator

import httpx

logger = logging.getLogger("asmr.danmaku")

NICK_KEYS = ("nickname", "nickName", "nick_name", "userNickname", "fromNickname", "displayName", "username", "userName")
TEXT_KEYS = ("content", "msgContent", "comment", "text", "msg", "message")
GIFT_KEYS = ("giftName", "gift_name", "rewardName", "reward_name", "giftTitle")
COUNT_KEYS = ("comboCount", "combo_count", "giftNum", "gift_num", "count", "num")
ID_KEYS = ("msgId", "msg_id", "clientMsgId", "client_msg_id", "seq", "id", "uuid")
USER_KEYS = ("user", "fromUser", "from_user", "contact", "sender", "author", "userInfo", "user_info")
JOIN_HINTS = ("进入直播间", "来了", "加入直播间")
FOLLOW_HINTS = ("关注了主播", "关注了你")


def _first(d: dict, keys: tuple[str, ...]):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", 0):
            return v
    return None


def _nickname(d: dict) -> str | None:
    nick = _first(d, NICK_KEYS)
    if isinstance(nick, str) and nick.strip():
        return nick.strip()
    for k in USER_KEYS:
        sub = d.get(k)
        if isinstance(sub, dict):
            nick = _first(sub, NICK_KEYS)
            if isinstance(nick, str) and nick.strip():
                return nick.strip()
    return None


def _gift(d: dict) -> str | None:
    g = _first(d, GIFT_KEYS)
    if isinstance(g, str):
        return g
    sub = d.get("gift") or d.get("giftInfo") or d.get("reward")
    if isinstance(sub, dict):
        g = _first(sub, ("name", "giftName", "title"))
        if isinstance(g, str):
            return g
    return None


def extract_events(value, platform: str) -> Iterator[dict]:
    """Yield normalized events found anywhere inside a decoded JSON value."""
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(reversed(v))
            continue
        if isinstance(v, str) and len(v) > 20 and v.lstrip()[:1] in "{[":
            try:  # JSON carried as a string (e.g. an HTTP body inside a WebSocket frame)
                stack.append(json.loads(v))
            except ValueError:
                pass
            continue
        if not isinstance(v, dict):
            continue
        nick = _nickname(v)
        gift = _gift(v)
        text = _first(v, TEXT_KEYS)
        if nick and (gift or isinstance(text, str)):
            ev = {"user": nick, "platform": platform}
            mid = _first(v, ID_KEYS)
            if mid is not None:
                ev["id"] = f"{platform}:{mid}"
            if gift:
                count = _first(v, COUNT_KEYS)
                ev.update(kind="gift", gift=gift, count=int(count) if str(count).isdigit() else 1)
            else:
                text = re.sub(r"\s+", " ", text).strip()
                if any(h in text for h in JOIN_HINTS) and len(text) <= 12:
                    ev.update(kind="join")
                elif any(h in text for h in FOLLOW_HINTS):
                    ev.update(kind="follow")
                else:
                    ev.update(kind="comment", text=text)
            yield ev
            continue  # don't also yield the same message's nested parts
        stack.extend(x for x in v.values() if isinstance(x, (dict, list, str)))


class StageSink:
    """Posts events to the stage server's /api/event, never blocking the capture."""

    def __init__(self, stage: str) -> None:
        self.url = stage.rstrip("/") + "/api/event"
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)
        self.task: asyncio.Task | None = None

    def put(self, ev: dict) -> None:
        try:
            self.queue.put_nowait(ev)
        except asyncio.QueueFull:
            logger.warning("event queue full, dropping %s", ev)

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                ev = await self.queue.get()
                for attempt in range(3):
                    try:
                        r = await client.post(self.url, json=ev)
                        res = r.json()
                        logger.info("%-7s %s: %s -> %s", ev["kind"], ev["user"], ev.get("text") or ev.get("gift") or "",
                                    res.get("action", "?") + (f" ({res['reason']})" if res.get("reason") else ""))
                        break
                    except Exception as e:  # noqa: BLE001 - stage restarting etc.
                        logger.warning("stage not reachable (%s), retrying", type(e).__name__)
                        await asyncio.sleep(1 + attempt * 2)

    def start(self) -> None:
        self.task = asyncio.ensure_future(self.run())
