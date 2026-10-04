"""Run the practice trainer: `uv run python -m stim.trainer --model runs/gen0/student.pt`."""

from __future__ import annotations

import argparse
import asyncio
import webbrowser
from pathlib import Path

from .server import run


def newest_student() -> str | None:
    found = sorted(Path("runs").glob("*/student*.pt"), key=lambda p: p.stat().st_mtime)
    return str(found[-1]) if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description="stim practice trainer")
    ap.add_argument("--model", default=None, help="student checkpoint (default: the newest runs/*/student*.pt)")
    ap.add_argument("--spec", default="feral")
    ap.add_argument("--scenario", default="boss")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    args = ap.parse_args()

    student = None
    path = args.model or newest_student()
    if path:
        import torch

        from ..student import load_student

        torch.set_num_threads(2)
        student = load_student(path)
        print(f"coach: {path}")
    else:
        print("no student checkpoint found; running without a coach")
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}/")
    try:
        asyncio.run(run(args.host, args.port, student, {"spec": args.spec, "scenario": args.scenario, "speed": args.speed}))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
