"""Full-fight DPS of policies on identical fight seeds, so differences are paired.

Policies: random, greedy, apl, student=<checkpoint>, addon=<checkpoint> (plays only on what the in-game
addon knows), and teacher:<rollout> where the rollout policy is greedy, apl, auto (the better
baseline), or student=<checkpoint>. With --randomize, every seed also
perturbs the spec's numbers (the same way for every policy), which tests whether a model reads the
numbers instead of memorizing them.

    uv run python scripts/evaluate.py --policies greedy apl student=runs/gen0/student.pt
    uv run python scripts/evaluate.py --policies apl teacher:apl --samples 16 --scenarios boss
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from stim.policies import AplPolicy, GreedyPolicy, RandomPolicy, best_baseline
from stim.scenarios import SCENARIOS, make_scenario
from stim.spec import available_specs, load_spec
from stim.teacher import Teacher, TeacherConfig, TeacherPolicy

_students: dict[str, object] = {}


def student(path: str):
    if path not in _students:
        import torch

        from stim.student import load_student

        torch.set_num_threads(1)
        _students[path] = load_student(path)
    return _students[path]


def make_policy(name: str, spec, args, seed: int):
    if name == "random":
        return RandomPolicy(random.Random(seed))
    if name == "greedy":
        return GreedyPolicy()
    if name == "apl":
        return AplPolicy(spec)
    if name == "auto":
        return best_baseline(spec)
    if name.startswith("student="):
        return student(name.split("=", 1)[1])
    if name.startswith("addon="):
        from stim.addon.policy import AddonPolicy, load_addon

        path = name.split("=", 1)[1]
        if path not in _students:
            import torch

            torch.set_num_threads(1)
            _students[path] = load_addon(path)[0]
        return AddonPolicy(_students[path])  # one per fight: it tracks that fight's casts
    if name.startswith("teacher:"):
        rollout = make_policy(name.split(":", 1)[1], spec, args, seed)
        return TeacherPolicy(Teacher(rollout, TeacherConfig(samples=args.samples, horizon=args.horizon)), random.Random(seed))
    raise ValueError(f"unknown policy {name!r}")


def play(task):
    spec_name, kind, policy_name, ep, args = task
    spec = load_spec(spec_name)
    if args.randomize > 0:
        spec = spec.randomized(random.Random(50_000 + ep), args.randomize)
    policy = make_policy(policy_name, spec, args, 7 + ep)
    sim = make_scenario(spec, kind, random.Random(args.seed + ep))
    lat = []
    while not sim.done:
        opts = sim.legal_options()
        t0 = time.perf_counter()
        o = policy(sim, opts)
        if len(opts) > 1:
            lat.append(time.perf_counter() - t0)
        sim.step(o)
    return spec_name, kind, policy_name, ep, sim.damage / sim.t, lat


def label(name: str) -> str:
    for kind in ("student", "addon"):
        if name.startswith(kind + "="):
            return kind + ":" + "/".join(name.split("=", 1)[1].split("/")[-2:])
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["greedy", "apl"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--specs", nargs="*", default=available_specs())
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--horizon", type=float, default=20.0)
    ap.add_argument("--randomize", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=900_000, help="fight seeds are seed + episode (held out from training)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    tasks = [(s, k, p, ep, args) for s in args.specs for k in args.scenarios for p in args.policies
             for ep in range(args.episodes)]
    tasks.sort(key=lambda t: not t[2].startswith("teacher"))  # slow ones first
    res: dict[tuple[str, str, str], dict[int, float]] = {}
    lat: dict[str, list[float]] = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for spec_name, kind, pol, ep, dps, l in pool.map(play, tasks, chunksize=1):
            res.setdefault((spec_name, kind, pol), {})[ep] = dps
            lat.setdefault(pol, []).extend(l)

    n = args.episodes
    base = [p for p in args.policies if p in ("random", "greedy", "apl", "auto")]
    print(f"{n} fights per cell, seeds {args.seed}+, randomize={args.randomize:g}; mean DPS ± standard error")
    print("Δ = paired difference to the best baseline in that row")
    for spec_name in args.specs:
        for kind in args.scenarios:
            row = f"{spec_name:13s} {kind:9s}"
            best = max(base, key=lambda p: statistics.mean(res[(spec_name, kind, p)].values())) if base else None
            for pol in args.policies:
                xs = [res[(spec_name, kind, pol)][e] for e in range(n)]
                row += f"  {label(pol)}={statistics.mean(xs):5.0f}±{statistics.stdev(xs) / n ** 0.5:3.0f}"
                if best and pol not in base:
                    d = [res[(spec_name, kind, pol)][e] - res[(spec_name, kind, best)][e] for e in range(n)]
                    row += f" (Δ{statistics.mean(d):+4.0f}±{statistics.stdev(d) / n ** 0.5:2.0f})"
            print(row)
    for pol, l in lat.items():
        if l and pol.startswith(("student", "addon")):
            a = np.array(l) * 1000
            print(f"{label(pol)} decision latency: p50 {np.percentile(a, 50):.2f} ms, p99 {np.percentile(a, 99):.2f} ms")


if __name__ == "__main__":
    main()
