"""Baseline policies. A policy maps (sim, legal options) to one option."""

from __future__ import annotations

import random
from types import SimpleNamespace

from .sim import POOL, Sim
from .valuation import option_value

Option = tuple[int, int]


class RandomPolicy:
    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random(0)

    def __call__(self, sim: Sim, opts: list[Option]) -> Option:
        return self.rng.choice(opts)


class GreedyPolicy:
    """Class-agnostic: picks the option with the best immediate expected value. No spec knowledge."""

    name = "greedy"

    def __call__(self, sim: Sim, opts: list[Option]) -> Option:
        best, best_v = opts[0], float("-inf")
        for o in opts:
            v = option_value(sim, o)
            if v > best_v:
                best, best_v = o, v
        return best


def apl_namespace(sim: Sim) -> dict:
    """Variables available to priority-list conditions, SimC-style (buff.x.up, dot.x.remains, ...)."""
    R = sim.R
    t = sim.t
    tgt = sim.enemy(sim.target)
    ns: dict = {}
    for i, r in enumerate(R.res):
        ns[r.id] = sim.points(i, tgt)
        ns[r.id + "_max"] = r.max
        ns[r.id + "_deficit"] = r.max - sim.res[i]
    buff = SimpleNamespace()
    for bi, b in enumerate(R.buffs):
        rem = max(0.0, sim.buff_until[bi] - t)
        setattr(buff, b.id, SimpleNamespace(up=rem > 0, remains=rem, stacks=sim.buff_stacks[bi] if rem > 0 else 0))
    debuff = SimpleNamespace()
    for di, d in enumerate(R.debuffs):
        cur = sim.debuff(tgt, di) if tgt is not None else None
        rem = cur[0] - t if cur else 0.0
        setattr(debuff, d.id, SimpleNamespace(up=rem > 0, remains=rem, stacks=cur[1] if cur else 0))
    dot = SimpleNamespace()
    cooldown = SimpleNamespace()
    for ai, a in enumerate(R.abilities):
        if a.dot is not None:
            cur = tgt.dots.get(ai) if tgt is not None else None
            rem = (cur[0] - t + (cur[1] - 1) * cur[3]) if cur else 0.0
            setattr(dot, a.id, SimpleNamespace(up=cur is not None, remains=rem, ticking=cur is not None))
        cd = max(0.0, sim.cd_until[ai] - t)
        setattr(cooldown, a.id, SimpleNamespace(ready=cd <= 0, remains=cd))
    near = sum(1 for e in sim.enemies if sim.dist(e) <= 8.0)
    ns.update(
        buff=buff, debuff=debuff, dot=dot, cooldown=cooldown, time=t,
        target=SimpleNamespace(
            health_pct=100.0 * tgt.hp / tgt.hp_max if tgt else 0.0,
            time_to_die=tgt.time_to_die if tgt else 0.0,
            distance=sim.dist(tgt) if tgt else 99.0,
        ),
        behind=sim.behind(tgt) if tgt else False,
        in_melee=tgt is not None and sim.dist(tgt) <= sim.spec.melee_range,
        enemies=near,
        active_enemies=len(sim.enemies),
    )
    return ns


class EpsilonPolicy:
    """Follows a policy but picks a uniformly random legal option with probability epsilon, so the
    states it visits include the mistakes a person makes."""

    def __init__(self, policy, epsilon: float, rng: random.Random):
        self.policy, self.epsilon, self.rng = policy, epsilon, rng
        self.name = f"{policy.name}+eps{epsilon:g}"

    def __call__(self, sim: Sim, opts: list[Option]) -> Option:
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(opts)
        return self.policy(sim, opts)


def best_baseline(spec, episodes: int = 8, seed: int = 0):
    """The better of the greedy and priority-list policies for this spec, by mean DPS over every
    scenario. Teachers roll out with it until a trained student replaces it."""
    from .scenarios import SCENARIOS, make_scenario

    best, best_dps = None, float("-inf")
    for policy in (GreedyPolicy(), AplPolicy(spec)):
        total = 0.0
        for kind in SCENARIOS:
            for ep in range(episodes):
                sim = make_scenario(spec, kind, random.Random(seed + ep))
                while not sim.done:
                    sim.step(policy(sim, sim.legal_options()))
                total += sim.damage / sim.t
        if total > best_dps:
            best, best_dps = policy, total
    return best


class AplPolicy:
    """A Hekili-style priority list from the spec's [[apl]] entries: the first usable entry wins."""

    name = "apl"

    def __init__(self, spec):
        self.entries = [
            (spec.ability_index(aid), compile(cond, f"apl:{aid}", "eval"), cond) for aid, cond in spec.apl
        ]

    def __call__(self, sim: Sim, opts: list[Option]) -> Option:
        if not self.entries:
            return (POOL, -1)
        by_ability: dict[int, list[Option]] = {}
        for o in opts:
            by_ability.setdefault(o[0], []).append(o)
        ns = None
        for ai, code, _ in self.entries:
            cands = by_ability.get(ai)
            if not cands:
                continue
            if ns is None:
                ns = apl_namespace(sim)
            if eval(code, {"__builtins__": {}}, ns):
                for o in cands:
                    if o[1] == sim.target:
                        return o
                return cands[0]
        return (POOL, -1)
