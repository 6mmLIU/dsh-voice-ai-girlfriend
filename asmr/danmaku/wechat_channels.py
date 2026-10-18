"""WeChat Channels (视频号) live comments, gifts and joins -> the stage.

    venv-asmr/bin/python -m asmr.danmaku.wechat_channels
    venv-asmr/bin/python -m asmr.danmaku.wechat_channels --dump      # also save raw data for tuning

Works the same on Windows, Linux and macOS: Playwright opens its own Chromium
window with a saved profile (asmr/out/channels-profile). The first time, scan
the QR code with WeChat; the login is remembered after that. Then go live from
the 视频号助手 live console in that window. Everything the page receives from
WeChat (XHR/fetch JSON and WebSocket frames) is scanned for viewer messages
and forwarded to POST /api/event on the stage; the stage decides what to
answer (asmr/interaction.py).

There is no public API for Channels live comments, so this reads what the
web console itself shows - the "browser mode" commercial tools use too.
--dom adds a fallback that reads newly added comment elements from the page.
--dump writes every JSON response and frame to asmr/out/channels-dump/, so
the parser can be tuned on real payloads (they change from time to time).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time

from ..config import OUT_DIR
from .common import StageSink, extract_events

logger = logging.getLogger("asmr.danmaku.channels")

PLATFORM = "wechat_channels"
START_URL = "https://channels.weixin.qq.com/platform/live/liveBuild"
HOSTS = ("weixin.qq.com", "wechat.com", "qq.com")

DOM_HOOK = r"""
(() => {
  if (window.__asmrDom) return;
  window.__asmrDom = true;
  const RE = /comment|message|msg|danmu|barrage|chat/i;
  const seen = new WeakSet();
  const scan = (node) => {
    if (!(node instanceof HTMLElement) || seen.has(node)) return;
    const cls = (node.className && node.className.toString()) || '';
    if (!RE.test(cls) || node.children.length > 6) return;
    seen.add(node);
    const text = (node.innerText || '').trim();
    if (text && text.length < 200 && window.asmrDomEvent) window.asmrDomEvent(cls.slice(0, 80), text);
  };
  new MutationObserver((muts) => {
    for (const m of muts) for (const n of m.addedNodes) {
      scan(n);
      if (n.querySelectorAll) n.querySelectorAll('*').forEach(scan);
    }
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
"""


class Capture:
    def __init__(self, sink: StageSink, dump: bool, dom: bool) -> None:
        self.sink = sink
        self.dump_dir = OUT_DIR / "channels-dump" if dump else None
        self.dom = dom
        self.n_dump = 0
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    def _dump(self, kind: str, url: str, data) -> None:
        if not self.dump_dir:
            return
        self.n_dump += 1
        path = self.dump_dir / f"{int(time.time())}_{self.n_dump:05d}_{kind}.json"
        path.write_text(json.dumps({"url": url, "data": data}, ensure_ascii=False, indent=1), encoding="utf-8")

    def _emit(self, data, source: str) -> None:
        for ev in extract_events(data, PLATFORM):
            ev["via"] = source
            self.sink.put(ev)

    async def on_response(self, response) -> None:
        try:
            if not any(h in response.url for h in HOSTS):
                return
            if response.request.resource_type not in ("xhr", "fetch"):
                return
            if "json" not in (response.headers.get("content-type") or ""):
                return
            data = await response.json()
        except Exception:  # noqa: BLE001 - bodies of redirects, aborted requests...
            return
        self._dump("http", response.url, data)
        self._emit(data, "http")

    def on_websocket(self, ws) -> None:
        logger.info("websocket opened: %s", ws.url[:100])

        def frame(payload) -> None:
            if isinstance(payload, (bytes, bytearray)):
                try:
                    payload = payload.decode("utf-8")
                except UnicodeDecodeError:
                    self._dump("ws_binary", ws.url, {"hex": payload[:2048].hex(), "len": len(payload)})
                    return
            try:
                data = json.loads(payload)
            except (TypeError, ValueError):
                self._dump("ws_text", ws.url, payload[:4000])
                return
            self._dump("ws", ws.url, data)
            self._emit(data, "ws")

        ws.on("framereceived", frame)

    def on_dom(self, _source, cls: str, text: str) -> None:
        """Fallback: a comment element appeared. Expect "昵称：内容" or "昵称\\n内容"."""
        m = re.match(r"^(.{1,20}?)\s*[:：\n]\s*(.+)$", text, re.S)
        if not m:
            return
        user, content = m.group(1).strip(), re.sub(r"\s+", " ", m.group(2)).strip()
        self._dump("dom", cls, {"text": text})
        self.sink.put({"kind": "comment", "user": user, "text": content, "platform": PLATFORM, "via": "dom"})

    async def attach(self, page) -> None:
        page.on("response", lambda r: asyncio.ensure_future(self.on_response(r)))
        page.on("websocket", self.on_websocket)


async def run(stage: str, dump: bool, dom: bool) -> None:
    from playwright.async_api import async_playwright

    sink = StageSink(stage)
    sink.start()
    cap = Capture(sink, dump, dom)
    profile = OUT_DIR / "channels-profile"
    profile.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            str(profile), headless=False, viewport=None, locale="zh-CN", args=["--start-maximized"])
        if dom:
            await ctx.expose_binding("asmrDomEvent", cap.on_dom)
            await ctx.add_init_script(DOM_HOOK)
        ctx.on("page", lambda p: asyncio.ensure_future(cap.attach(p)))
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await cap.attach(page)
        await page.goto(START_URL)
        if "login" in page.url:
            logger.info("请在弹出的浏览器窗口里用微信扫码登录视频号助手（只需第一次）")
            while "login" in page.url:
                await asyncio.sleep(1)
            logger.info("已登录")
        logger.info("在这个窗口里打开「直播」开播；评论、礼物、进场会自动转发到舞台 %s", stage)
        if dump:
            logger.info("原始数据会保存到 %s", cap.dump_dir)
        closed = asyncio.get_running_loop().create_future()
        ctx.on("close", lambda *_: closed.done() or closed.set_result(True))
        await closed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="http://127.0.0.1:8770", help="stage server address")
    parser.add_argument("--dump", action="store_true", help="save raw payloads to asmr/out/channels-dump/")
    parser.add_argument("--dom", action="store_true", help="also read comment elements from the page")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args.stage, args.dump, args.dom))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
