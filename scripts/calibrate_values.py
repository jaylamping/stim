"""Measure what each resource is worth under a policy and suggest the spec's [teacher] resource_value.

The teacher values resources still banked when a rollout ends at these exchange rates, so they should
match what the rollout policy actually gets out of a resource. For states sampled from the policy's
own play, two clones share a seed and a future; one gets extra resource. Both play on for --window
seconds, and the difference in expected damage per unit of resource is its marginal value.

    uv run python scripts/calibrate_values.py --policy apl
    uv run python scripts/calibrate_values.py --policy apl --write   # update specs/*.toml
"""

from __future__ import annotations

import argparse
import os
import random
import re
import statistics
from concurrent.futures import ProcessPoolExecutor

from stim.policies import AplPolicy, GreedyPolicy
from stim.scenarios import SCENARIOS, make_scenario
from stim.spec import SPECS_DIR, available_specs, load_spec
from stim.teacher import fight_left


def make_policy(name: str, spec):
    return AplPolicy(spec) if name == "apl" else GreedyPolicy()


def extra_amount(res) -> float:
    if res.kind == "secondary":
        return 1.0
    if res.kind == "mana":
        return round(res.max * 0.1)
    return 20.0


def play_out(sim, policy) -> float:
    while not sim.done:
        sim.step(policy(sim, sim.legal_options()))
    return sim.ev_damage


def measure(task) -> dict[str, list[float]]:
    spec_name, kind, policy_name, ep, window, every = task
    spec = load_spec(spec_name)
    policy = make_policy(policy_name, spec)
    rng = random.Random(10_000 + ep)
    sim = make_scenario(spec, kind, random.Random(rng.getrandbits(32)))
    out: dict[str, list[float]] = {r.id: [] for r in spec.resources}
    steps = 0
    while not sim.done:
        sim.step(policy(sim, sim.legal_options()))
        steps += 1
        if sim.done or steps % every or sim.t < 8.0 or fight_left(sim, sim.t_max) < window + 5.0:
            continue
        for ri, res in enumerate(spec.resources):
            delta = extra_amount(res)
            if sim.res[ri] + delta > res.max:
                continue
            target = sim.enemy(sim.target)
            if res.on_target and (target is None or sim.res_target[ri] not in (-1, target.id)):
                continue
            seed = rng.getrandbits(62)
            end = min(sim.t_max, sim.t + window)
            events = sim.plan.sample(random.Random(rng.getrandbits(62)), sim.t, end, sim.last_event)
            base, more = sim.clone(seed), sim.clone(seed)
            for c in (base, more):
                c.events, c.ev_i, c.t_max = events, 0, end
            more.res[ri] += delta
            if res.on_target:
                more.res_target[ri] = target.id
            out[res.id].append((play_out(more, policy) - play_out(base, policy)) / delta)
    return out


def suggest(x: float) -> float:
    """Two significant figures."""
    if x <= 0:
        return 0.0
    digits = 1 - int(f"{x:e}".split("e")[1])
    return round(x, digits) if digits > 0 else float(round(x, digits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="*", default=available_specs())
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--policy", choices=["apl", "greedy"], default="apl")
    ap.add_argument("--episodes", type=int, default=150, help="per spec and scenario")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--every", type=int, default=6, help="sample one state every N decisions")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--write", action="store_true", help="write the suggestions into specs/*.toml")
    args = ap.parse_args()

    tasks = [(s, k, args.policy, ep, args.window, args.every) for s in args.specs for k in args.scenarios
             for ep in range(args.episodes)]
    samples: dict[str, dict[str, list[float]]] = {s: {} for s in args.specs}
    with ProcessPoolExecutor(args.workers) as pool:
        for task, out in zip(tasks, pool.map(measure, tasks, chunksize=4)):
            for rid, xs in out.items():
                samples[task[0]].setdefault(rid, []).extend(xs)

    for spec_name in args.specs:
        spec = load_spec(spec_name)
        values = dict(spec.resource_value)
        print(f"{spec.name} under {args.policy} ({args.window:g} s window):")
        for rid, xs in samples[spec_name].items():
            if len(xs) < 10:
                print(f"  {rid:14s} too few samples ({len(xs)})")
                continue
            m, se = statistics.mean(xs), statistics.stdev(xs) / len(xs) ** 0.5
            values[rid] = suggest(m)
            print(f"  {rid:14s} {m:8.2f} ± {se:5.2f}  (n={len(xs)}, spec has {spec.resource_value.get(rid, 0):g})")
        line = "resource_value = { " + ", ".join(f"{k} = {v:g}" for k, v in values.items()) + " }"
        print(f"  suggested: {line}")
        if args.write:
            path = SPECS_DIR / f"{spec_name}.toml"
            text, n = re.subn(r"^resource_value = \{[^}\n]*\}$", line, path.read_text(), count=1, flags=re.M)
            if n != 1:
                raise SystemExit(f"{path}: no resource_value line to replace")
            path.write_text(text)
            print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
