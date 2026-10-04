"""Encounter generators. Each returns a fresh Sim with enemies placed and disruptions scheduled."""

from __future__ import annotations

import math
import random

from .sim import TANK_GAP, EventPlan, Sim
from .spec import Spec

SCENARIOS = ("boss", "boss_adds", "trash")


def make_scenario(spec: Spec, kind: str, rng: random.Random) -> Sim:
    if kind == "boss":
        plan = EventPlan(knockback=(18.0, 35.0), turn=(12.0, 25.0), reposition=(20.0, 40.0))
        ttd = rng.uniform(45.0, 120.0)
        party = rng.uniform(1800.0, 3200.0)
        sim = Sim(spec, rng, t_max=ttd * 1.6 + 20.0, plan=plan)
        sim.add_enemy("Boss", 0.0, TANK_GAP, (party + spec.reference_dps) * ttd, party, boss=True)
    elif kind == "boss_adds":
        plan = EventPlan(
            knockback=(25.0, 45.0), turn=(15.0, 30.0), wave=(18.0, 32.0),
            wave_size=(2, 3), add_hp=(12000.0, 30000.0), add_party_dps=(900.0, 1700.0),
        )
        ttd = rng.uniform(60.0, 120.0)
        party = rng.uniform(1500.0, 2600.0)
        sim = Sim(spec, rng, t_max=ttd * 1.6 + 20.0, plan=plan)
        sim.add_enemy("Boss", 0.0, TANK_GAP, (party + spec.reference_dps) * ttd, party, boss=True)
    elif kind == "trash":
        plan = EventPlan(reposition=(12.0, 25.0), reposition_distance=(4.0, 10.0))
        sim = Sim(spec, rng, t_max=90.0, plan=plan)
        base = rng.uniform(0, 2 * math.pi)
        for i in range(rng.randint(3, 6)):
            ang = base + rng.uniform(-0.9, 0.9)
            r = rng.uniform(8.0, 14.0)
            sim.add_enemy(
                f"Mob {i + 1}", math.cos(ang) * r, math.sin(ang) * r,
                rng.uniform(12000.0, 45000.0), rng.uniform(1100.0, 2300.0),
            )
    else:
        raise ValueError(f"unknown scenario {kind!r}; choose from {SCENARIOS}")
    sim.schedule(plan.sample(rng, 0.0, sim.t_max))
    return sim
