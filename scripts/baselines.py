"""Average DPS of the baseline policies (random, greedy, priority list) for every spec and scenario."""

import random, time, statistics
from stim.spec import load_spec, available_specs
from stim.scenarios import make_scenario, SCENARIOS
from stim.policies import RandomPolicy, GreedyPolicy, AplPolicy

import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for name in available_specs():
    spec = load_spec(name)
    pols = [RandomPolicy(random.Random(7)), GreedyPolicy(), AplPolicy(spec)]
    for kind in SCENARIOS:
        row = []
        for pol in pols:
            dps, t0, steps = [], time.perf_counter(), 0
            for ep in range(N):
                sim = make_scenario(spec, kind, random.Random(1000 + ep))
                while not sim.done:
                    sim.step(pol(sim, sim.legal_options())); steps += 1
                dps.append(sim.damage / sim.t)
            m = statistics.mean(dps); se = statistics.stdev(dps) / N**0.5
            row.append(f"{pol.name}={m:6.0f}±{se:3.0f} ({steps/(time.perf_counter()-t0)/1000:4.0f}k/s)")
        print(f"{spec.name:14s} {kind:10s} " + "  ".join(row))
