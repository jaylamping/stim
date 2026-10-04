"""Practice-trainer server: serves the page and runs one fight per WebSocket connection.

The browser sends key presses and target clicks; the server runs the fight on a real clock (scaled by
the chosen speed), streams the state 20 times a second with the model's advice, and sends a review
when the pull ends.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from ..scenarios import SCENARIOS
from ..spec import available_specs, load_spec
from .session import Session

STATIC = Path(__file__).parent / "static"
FPS = 20


def page(connection: ServerConnection, request: Request) -> Response | None:
    if request.path.startswith("/ws"):
        return None  # let the WebSocket handshake proceed
    if request.path in ("/", "/index.html"):
        body = (STATIC / "index.html").read_bytes()
        return Response(200, "OK", Headers({"Content-Type": "text/html; charset=utf-8",
                                            "Content-Length": str(len(body)), "Cache-Control": "no-store"}), body)
    return Response(404, "Not Found", Headers({"Content-Length": "0"}), b"")


def make_handler(student, defaults: dict):
    async def handler(ws: ServerConnection) -> None:
        specs = [{"id": s, "name": load_spec(s).name} for s in available_specs()]
        await ws.send(json.dumps({"type": "hello", "specs": specs, "scenarios": list(SCENARIOS),
                                  "defaults": defaults, "model": student is not None}))
        session: Session | None = None

        async def review(s: Session) -> None:
            await ws.send(json.dumps({"type": "reviewing"}))
            model_dps = await asyncio.to_thread(s.model_run)
            graded = await asyncio.to_thread(s.review)
            await ws.send(json.dumps({"type": "review", "dps": s.sim.damage / max(s.sim.t, 1e-9),
                                      "model_dps": model_dps, **graded}))

        async def ticker() -> None:
            last = time.perf_counter()
            while True:
                await asyncio.sleep(1 / FPS)
                now = time.perf_counter()
                dt, last = min(now - last, 0.25), now
                s = session
                if s is None:
                    continue
                was_done = s.sim.done
                s.tick(dt)
                await ws.send(json.dumps({"type": "state", **s.state()}))
                if s.sim.done and not was_done:
                    asyncio.create_task(review(s))

        task = asyncio.create_task(ticker())
        try:
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "start":
                    session = Session(msg.get("spec", defaults["spec"]), msg.get("scenario", defaults["scenario"]),
                                      msg.get("seed"), float(msg.get("speed", defaults["speed"])), student)
                elif session is None:
                    continue
                elif kind == "press":
                    session.press(str(msg.get("key", "")))
                elif kind == "target":
                    session.target(int(msg.get("id", -1)))
                elif kind == "cycle":
                    session.cycle_target()
                elif kind == "pause":
                    session.paused = bool(msg.get("value", not session.paused))
                elif kind == "speed":
                    session.speed = min(2.0, max(0.1, float(msg.get("value", 1.0))))
        except ConnectionClosed:
            pass
        finally:
            task.cancel()

    return handler


async def run(host: str, port: int, student, defaults: dict) -> None:
    async with serve(make_handler(student, defaults), host, port, process_request=page) as server:
        print(f"stim trainer on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
        await server.serve_forever()
