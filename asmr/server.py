"""ASMR live stage server.

    venv-asmr/bin/python -m asmr.server

  /                 stage page: full-screen 9:16, capture this window (and its
                    audio) in the Douyin / Kuaishou live companion or OBS
  /control          operator panel: simulated danmaku, manual cues, log
  POST /api/danmaku {user, text}   a viewer message (a real danmaku fetcher
                    posts here too)
  POST /api/cue     {script}       play a hand-written tagged script next
  POST /api/perform {key, pos, seconds | forever}   start / switch an action now
  POST /api/stop                   stop the current action
  POST /api/length  {seconds | forever}             change the current action's length
  GET  /api/now                    what is being performed right now
  WS   /ws          units + log to the stage and control pages; the stage
                    reports {"type":"started","id"} for every unit it starts

The performer (performer.py) streams short units to the stage one at a time,
so every action can be stopped, switched or extended live. When the chat is
quiet an idle program keeps the stream alive.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import actions, media
from .config import CONFIG, MEDIA_DIR, OUT_DIR, STAGE_DIR
from .director import Director
from .interaction import Interaction
from .performer import Performer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("asmr.server")

MAX_PENDING = 5  # danmaku waiting to be answered; older ones are dropped

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    tasks = [asyncio.create_task(producer()), asyncio.create_task(performer.run())]
    backend = CONFIG.get("tts", {}).get("backend", "say")
    llm = "on" if director.has_llm else "off (fallback lines; set DEEPSEEK_API_KEY)"
    host, port = CONFIG.get("host", "127.0.0.1"), CONFIG.get("port", 8770)
    logger.info("ASMR stage ready: http://%s:%s/  control: http://%s:%s/control  tts=%s llm=%s",
                host, port, host, port, backend, llm)
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="ASMR stage", lifespan=lifespan)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/out", StaticFiles(directory=OUT_DIR), name="out")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/stage", StaticFiles(directory=STAGE_DIR), name="stage")


@app.middleware("http")
async def no_stale_pages(request: Request, call_next):
    """Stage/control pages must pick up code changes on a plain reload."""
    response = await call_next(request)
    if request.url.path.startswith("/stage/") or request.url.path == "/control":
        response.headers["Cache-Control"] = "no-cache"
    return response

director = Director()
clients: set[WebSocket] = set()
danmaku: asyncio.Queue[dict] = asyncio.Queue()
state = {"rendering": False}


async def broadcast(event: dict) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            clients.discard(ws)


async def log(text: str) -> None:
    logger.info(text)
    await broadcast({"type": "log", "text": text, "at": time.time()})


performer = Performer(broadcast, log)


async def producer() -> None:
    """Answer danmaku one by one; each reply joins the performer's queue."""
    while True:
        msg = await danmaku.get()
        state["rendering"] = True
        try:
            script = await director.reply(msg["user"], msg["text"])
            await log(f"导演 → {msg['user']}：{script}")
            performer.enqueue_script(script, {"user": msg["user"], "danmaku": msg["text"]})
        except Exception as exc:  # noqa: BLE001
            logger.exception("reply failed")
            await log(f"回复失败：{type(exc).__name__}: {exc}")
        finally:
            state["rendering"] = False


@app.get("/")
async def index(request: Request) -> RedirectResponse:
    query = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(f"/stage/index.html{query}")


@app.get("/control")
async def control() -> FileResponse:
    return FileResponse(STAGE_DIR / "control.html")


class DanmakuIn(BaseModel):
    user: str = "观众"
    text: str


@app.post("/api/danmaku")
async def post_danmaku(body: DanmakuIn) -> dict:
    while danmaku.qsize() >= MAX_PENDING:
        dropped = danmaku.get_nowait()
        await log(f"弹幕太多，跳过：{dropped['user']}：{dropped['text']}")
    msg = {"user": body.user.strip() or "观众", "text": body.text.strip()}
    await danmaku.put(msg)
    await broadcast({"type": "danmaku", **msg})
    return {"ok": True, "pending": danmaku.qsize()}


class EventIn(BaseModel):
    kind: str = "comment"  # comment | gift | join | follow | like
    user: str = "观众"
    text: str = ""
    gift: str = ""
    count: int = 1
    id: str = ""
    platform: str = ""


interaction = Interaction()


@app.post("/api/event")
async def post_event(body: EventIn) -> dict:
    """Normalized viewer events from platform adapters (asmr/interaction.py decides what to do)."""
    busy = danmaku.qsize() > 0 or state["rendering"] or len(performer.steps) > 0
    d = interaction.decide(body.model_dump(), busy)
    if d.action == "reply":
        return {**await post_danmaku(DanmakuIn(user=body.user, text=body.text)), "action": "reply"}
    if d.action == "script":
        performer.enqueue_script(d.script, {"source": body.kind, "user": body.user}, front=d.front)
        await log(f"{body.platform or '直播间'} {body.kind} {body.user}：{d.script}")
        return {"ok": True, "action": "script"}
    return {"ok": True, "action": "drop", "reason": d.reason}


class CueIn(BaseModel):
    script: str


@app.post("/api/cue")
async def post_cue(body: CueIn) -> dict:
    performer.enqueue_script(body.script, {"source": "manual"}, front=True)
    await log(f"中控插播：{body.script}")
    return {"ok": True}


class PerformIn(BaseModel):
    key: str
    pos: str = "C"  # L / R / C / SWEEP
    seconds: float | None = None  # None = the action's configured length
    forever: bool = False  # keep going until stopped


@app.post("/api/perform")
async def post_perform(body: PerformIn) -> dict:
    if body.key not in actions.BY_KEY:
        return {"ok": False, "error": f"unknown action {body.key}"}
    performer.perform(body.key, body.pos, body.seconds, body.forever)
    await log(f"中控：开始 {actions.BY_KEY[body.key].tag}" + ("（一直做）" if body.forever else ""))
    return {"ok": True, **performer.status()}


@app.post("/api/stop")
async def post_stop() -> dict:
    performer.stop()
    await log("中控：停止当前动作")
    return {"ok": True, **performer.status()}


class LengthIn(BaseModel):
    seconds: float | None = None
    forever: bool = False


@app.post("/api/length")
async def post_length(body: LengthIn) -> dict:
    performer.set_length(None if body.forever else body.seconds)
    return {"ok": True, **performer.status()}


@app.get("/api/now")
async def get_now() -> dict:
    return performer.status()


@app.get("/api/status")
async def status() -> dict:
    return {
        **performer.status(),
        "pending_danmaku": danmaku.qsize() + (1 if state["rendering"] else 0),
        "llm": director.has_llm,
        "tts": CONFIG.get("tts", {}).get("backend", "say"),
    }


@app.get("/api/media")
async def media_list() -> dict:
    """Today's outfit, its video clips per action and idle stills (asmr/media.py)."""
    return media.listing()


@app.get("/api/actions")
async def action_list() -> dict:
    """The action catalog with current durations (seconds)."""
    return actions.listing()


class ActionIn(BaseModel):
    key: str | None = None
    seconds: float | None = None  # None with a key = back to the default
    scale: float | None = None  # global multiplier for every action


@app.post("/api/actions")
async def action_set(body: ActionIn) -> dict:
    actions.set_seconds(body.key, body.seconds, body.scale)
    return actions.listing()


@app.get("/api/outfits")
async def outfit_list() -> dict:
    return {"outfits": media.outfits(), "current": media.current_outfit(), "pinned": media.pinned_outfit()}


class OutfitIn(BaseModel):
    name: str | None = None  # None = back to the daily rotation


@app.post("/api/outfit")
async def outfit_set(body: OutfitIn) -> dict:
    media.pin_outfit(body.name if body.name in media.outfits() else None)
    await log(f"换装：{media.current_outfit()}（{'固定' if media.pinned_outfit() else '按天轮换'}）")
    await broadcast({"type": "outfit", "outfit": media.current_outfit()})
    return await outfit_list()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            text = await ws.receive_text()
            if text.startswith("{"):
                msg = json.loads(text)
                if msg.get("type") == "started":
                    performer.on_started(msg.get("id", ""))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


def main() -> None:
    uvicorn.run(app, host=CONFIG.get("host", "127.0.0.1"), port=int(CONFIG.get("port", 8770)), log_level="warning")


if __name__ == "__main__":
    main()
