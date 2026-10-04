"""Generate teacher-labeled decision states with parallel workers.

Each worker plays a chunk of episodes and writes one shard (`runs/<name>/shard_*.pkl`). Episodes per
spec and scenario are set by --episodes; every episode gets its own seed and randomized spec.

    uv run python scripts/gen_data.py --out runs/gen0 --episodes 300
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stim.data import GenConfig, generate_episode, save_shard
from stim.scenarios import SCENARIOS
from stim.spec import available_specs


def work(job) -> tuple[int, int, float]:
    shard, tasks, cfg, out = job
    start = time.perf_counter()
    episodes = [generate_episode(spec_name, kind, seed, cfg) for spec_name, kind, seed in tasks]
    save_shard(Path(out) / f"shard_{shard:05d}.pkl", episodes)
    return len(episodes), sum(len(e.records) for e in episodes), time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100, help="per spec and scenario")
    ap.add_argument("--specs", nargs="*", default=available_specs())
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8, help="episodes per shard")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    for f in dataclasses.fields(GenConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg = GenConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(GenConfig)})

    out = Path(args.out)
    if any(out.glob("shard_*.pkl")):
        raise SystemExit(f"{out} already has shards; pick a new --out")
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for s in args.specs:
        for k in args.scenarios:
            for ep in range(args.episodes):
                tasks.append((s, k, args.seed * 1_000_003 + len(tasks)))
    # Interleave specs and scenarios so every shard is a mix and partial runs stay balanced.
    tasks = tasks[::3] + tasks[1::3] + tasks[2::3]
    jobs = [(i, tasks[j : j + args.chunk], cfg, str(out)) for i, j in enumerate(range(0, len(tasks), args.chunk))]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"config": dataclasses.asdict(cfg), "specs": args.specs, "scenarios": args.scenarios,
            "episodes_per": args.episodes, "seed": args.seed, "commit": commit, "started": time.ctime()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    start = time.perf_counter()
    done_eps = done_states = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            eps, states, _ = fut.result()
            done_eps += eps
            done_states += states
            elapsed = time.perf_counter() - start
            eta = elapsed / done_eps * (len(tasks) - done_eps)
            print(f"[{i}/{len(jobs)} shards] {done_eps} episodes, {done_states} states, "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
    meta.update(finished=time.ctime(), episodes=done_eps, states=done_states, seconds=round(time.perf_counter() - start))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {done_states} labeled states from {done_eps} episodes to {out}")


if __name__ == "__main__":
    main()
