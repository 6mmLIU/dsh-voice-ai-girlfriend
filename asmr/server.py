"""ASMR live stage server.

    venv-asmr/bin/python -m asmr.server

  /                 stage page: full-screen 9:16, capture this window (and its
                    audio) in the Douyin / Kuaishou live companion or OBS
  /control          operator panel: simulated danmaku, manual cues, log
  POST /api/danmaku {user, text}   a viewer message (a real danmaku fetcher
                    posts here too)
  POST /api/cue     {script}       play a hand-written tagged script now
  WS   /ws          play events + log, pushed to stage and control pages

The server owns the playback clock: one item plays at a time, the next reply
renders while the current one plays, and when the chat is quiet an idle
program (felt, taps, lights, the odd whisper) keeps the stream alive.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import actions, media
from .config import CONFIG, MEDIA_DIR, OUT_DIR, STAGE_DIR
from .director import Director, idle_script
from .render import render_script

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("asmr.server")

MAX_PENDING = 5  # danmaku waiting to be answered; older ones are dropped

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    tasks = [asyncio.create_task(producer()), asyncio.create_task(consumer())]
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
ready: asyncio.Queue[dict] = asyncio.Queue(maxsize=2)
manual: asyncio.Queue[dict] = asyncio.Queue()
state = {"playing": None, "until": 0.0, "rendering": False}


async def broadcast(event: dict) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            clients.discard(ws)


async def log(text: str) -> None:
    logger.info(text)
    await broadcast({"type": "log", "text": text, "at": time.time()})


async def producer() -> None:
    """Answer danmaku one by one; the rendered item waits in `ready`."""
    while True:
        msg = await danmaku.get()
        state["rendering"] = True
        try:
            script = await director.reply(msg["user"], msg["text"])
            await log(f"导演 → {msg['user']}：{script}")
            item = await render_script(script, {"user": msg["user"], "danmaku": msg["text"]})
            await ready.put(item)
        except Exception as exc:  # noqa: BLE001
            logger.exception("reply failed")
            await log(f"回复失败：{type(exc).__name__}: {exc}")
        finally:
            state["rendering"] = False


async def consumer() -> None:
    """Play one item at a time: manual cues first, then replies, else idle."""
    idle_after = float(CONFIG.get("idle", {}).get("after_sec", 12))
    while True:
        item = None
        if not manual.empty():
            item = manual.get_nowait()
        elif not ready.empty():
            item = ready.get_nowait()
        else:
            getter = asyncio.ensure_future(ready.get())
            waiter = asyncio.ensure_future(manual.get())
            done, pending = await asyncio.wait({getter, waiter}, timeout=idle_after, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            # Both may finish together: play one, put the other back in front.
            finished = [t.result() for t in (waiter, getter) if t in done]
            if finished:
                item = finished[0]
                for extra in finished[1:]:
                    await (manual if extra.get("source") == "manual" else ready).put(extra)
        if item is None:
            if state["rendering"] or danmaku.qsize() > 0:
                continue  # a reply is on its way; don't cover it with idle
            try:
                item = await render_script(idle_script(), {"source": "idle"})
            except Exception:  # noqa: BLE001
                logger.exception("idle render failed")
                await asyncio.sleep(2)
                continue
        state["playing"] = item
        state["until"] = time.time() + item["duration"]
        await broadcast(item)
        await log(f"播放 {item.get('source', 'reply')} {item['duration']:.1f}s：{item['display'] or item['script']}")
        await asyncio.sleep(item["duration"] + 0.25)
        state["playing"] = None


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


class CueIn(BaseModel):
    script: str


@app.post("/api/cue")
async def post_cue(body: CueIn) -> dict:
    item = await render_script(body.script, {"source": "manual"})
    await manual.put(item)
    await log(f"中控插播：{body.script}")
    return {"ok": True, "id": item["id"], "duration": item["duration"]}


@app.get("/api/status")
async def status() -> dict:
    playing = state["playing"]
    return {
        "playing": playing and {k: playing[k] for k in ("id", "display", "duration")},
        "remaining": max(0.0, state["until"] - time.time()) if playing else 0.0,
        "pending_danmaku": danmaku.qsize(),
        "ready": ready.qsize(),
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
            await ws.receive_text()  # keepalive pings from the page
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


def main() -> None:
    uvicorn.run(app, host=CONFIG.get("host", "127.0.0.1"), port=int(CONFIG.get("port", 8770)), log_level="warning")


if __name__ == "__main__":
    main()
