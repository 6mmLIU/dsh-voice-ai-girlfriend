"""A fake live room: posts random comments, gifts, joins, follows and likes to the stage.

    venv-asmr/bin/python -m asmr.danmaku.mock                 # one event every ~6 s
    venv-asmr/bin/python -m asmr.danmaku.mock --every 2 --count 20

Use it to rehearse a stream (and to test the interaction rules) without going live.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random

from .common import StageSink

USERS = ["月亮不睡", "小鹿", "加班狗", "晚风", "橘子汽水", "失眠星人", "阿哲", "团团", "南风知我意", "一只猫"]
COMMENTS = ["晚上好", "今天好累啊", "能叫一下我的名字吗", "睡不着", "声音好好听", "刚下班", "想听敲击声", "想听掏耳朵",
            "明天要考试了好紧张", "左耳好舒服", "陪我待一会儿", "可以刷刷耳朵吗", "毛绒的声音好治愈", "晚安"]
GIFTS = ["小心心", "玫瑰", "星星", "比心"]
WEIGHTS = {"comment": 6, "join": 3, "gift": 1, "like": 2, "follow": 1}


def fake_event(i: int) -> dict:
    kind = random.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
    ev = {"kind": kind, "user": random.choice(USERS), "id": f"mock:{i}", "platform": "mock"}
    if kind == "comment":
        ev["text"] = random.choice(COMMENTS)
    elif kind == "gift":
        ev.update(gift=random.choice(GIFTS), count=random.choice([1, 1, 1, 5, 10]))
    elif kind == "like":
        ev["count"] = random.randint(1, 20)
    return ev


async def run(stage: str, every: float, count: int) -> None:
    sink = StageSink(stage)
    sink.start()
    i = 0
    while count <= 0 or i < count:
        sink.put(fake_event(i))
        i += 1
        await asyncio.sleep(random.uniform(every * 0.5, every * 1.5))
    await asyncio.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="http://127.0.0.1:8770")
    parser.add_argument("--every", type=float, default=6.0, help="average seconds between events")
    parser.add_argument("--count", type=int, default=0, help="stop after this many events (0 = forever)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args.stage, args.every, args.count))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
