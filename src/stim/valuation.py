"""Expected-value estimates of options: the "tooltip math" shared by the greedy policy and the model's features."""

from __future__ import annotations

import math
import weakref

from .sim import POOL, Enemy, Sim

# Rough share of a melee spec's damage by school, used to value "damage taken" debuffs.
SCHOOL_SHARE = {"physical": 0.75, "bleed": 0.15}
WHITE_SHARE = 0.45


def expected_strike(sim: Sim, ai: int, e: Enemy | None) -> tuple[float, int]:
    """Expected direct damage of an ability summed over every target it would hit, and the target count."""
    R = sim.R
    a = R.abilities[ai]
    dmg = a.damage
    if dmg is None:
        return 0.0, 0
    if dmg.aoe_radius > 0:
        targets = sim._aoe_targets(dmg)
    elif e is None:
        return 0.0, 0
    else:
        targets = [e] + sim._cleave_targets(e, dmg.extra_targets)
    fin = R.finisher[ai]
    points = sim.points(fin, e) if fin >= 0 else 0.0
    base = dmg.weapon * sim.spec.main_hand.damage + dmg.flat
    if dmg.by_points:
        base += dmg.by_points[min(int(points), len(dmg.by_points) - 1)]
    if R.extra[ai] is not None:
        r, mx, per = R.extra[ai]
        paid = sum(amt for rr, amt in sim.cost_of(ai) if rr == r)
        base += min(max(0.0, sim.res[r] - paid), mx) * per
    sp = sim.spec
    hit = 1.0 - sp.miss_chance - sp.dodge_chance
    critf = 1.0 + sim.crit_chance(dmg.crit_bonus) * (sp.crit_multiplier - 1.0) if dmg.can_crit else 1.0
    total = 0.0
    for x in targets:
        total += min(x.hp, base * hit * critf * sim.damage_mult(x, dmg.school))
    return total, len(targets)


def expected_dot(sim: Sim, ai: int, e: Enemy | None) -> tuple[float, float]:
    """(damage the DoT would deal before the target dies, damage of the existing DoT lost by refreshing)."""
    R = sim.R
    a = R.abilities[ai]
    if a.dot is None or e is None:
        return 0.0, 0.0
    fin = R.finisher[ai]
    points = sim.points(fin, e) if fin >= 0 else 0.0
    tick = (a.dot.per_tick + a.dot.per_tick_per_point * points) * sim.damage_mult(e, a.dot.school)
    ttd = e.time_to_die
    n = min(a.dot.ticks, int(ttd / a.dot.interval))
    lost = 0.0
    cur = e.dots.get(ai)
    if cur is not None:
        nt, left, dmg, iv, _ = cur
        alive_ticks = max(0, int((ttd - (nt - sim.t)) / iv) + 1)
        lost = min(left, alive_ticks) * dmg
    return n * tick, lost


def resource_rate(sim: Sim, r: int) -> float:
    res = sim.R.res[r]
    if res.tick_interval > 0:
        return res.tick_amount / res.tick_interval
    if res.regen_per_sec > 0:
        return res.regen_per_sec
    if res.per_white_damage > 0:
        return res.per_white_damage * sim.spec.reference_dps * WHITE_SHARE
    return 0.0


def buff_rate(sim: Sim, bi: int, costs: bool = True) -> float:
    """Damage-equivalent value per second of a buff being up (costs=False leaves out cost reductions)."""
    b = sim.R.buffs[bi]
    ref = sim.spec.reference_dps
    v = ref * b.damage_mult
    v += ref * WHITE_SHARE * b.haste
    v += ref * b.crit_bonus * (sim.spec.crit_multiplier - 1.0)
    values = sim.spec.resource_value
    if costs:
        for k, m in b.cost_mult.items():
            rids = range(sim.R.n_res) if k == "all" else [sim.R.res_index[k]]
            for r in rids:
                v += resource_rate(sim, r) * (1.0 - m) * values.get(sim.R.res_ids[r], 0.0)
    for k, m in b.regen_mult.items():
        r = sim.R.res_index[k]
        v += resource_rate(sim, r) * (m - 1.0) * values.get(k, 0.0)
    if b.cleave_targets and len(sim.enemies) > 1:
        v += ref * 0.6 * min(b.cleave_targets, len(sim.enemies) - 1)
    return v


def buff_value(sim: Sim, bi: int, points: float | None) -> float:
    b = sim.R.buffs[bi]
    dur = b.duration
    if b.duration_by_points and points is not None:
        dur = b.duration_by_points[min(int(points), len(b.duration_by_points) - 1)]
    remaining = max(0.0, sim.buff_until[bi] - sim.t)
    cover = max(0.0, dur - remaining)
    return buff_rate(sim, bi) * cover


_upkeep_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def aura_upkeep(sim: Sim) -> tuple[list[float], list[float]]:
    """(buffs, debuffs): the cheapest cost per second, in damage, of keeping each aura up by reapplying
    it: resources and points spent at the spec's exchange rates. GCD time counts as free, since melee
    rotations are resource-bound and pool through spare GCDs. Infinite for auras no ability can keep
    up (procs, or a cooldown at least as long as the aura)."""
    R = sim.R
    cached = _upkeep_cache.get(R)
    if cached is not None:
        return cached
    sp = sim.spec
    values = sp.resource_value
    buffs = [math.inf] * len(R.buffs)
    debuffs = [math.inf] * len(R.debuffs)
    for ai, a in enumerate(R.abilities):
        for idx, auras, out in ((R.buff_of[ai], R.buffs, buffs), (R.debuff_of[ai], R.debuffs, debuffs)):
            if idx < 0:
                continue
            aura = auras[idx]
            base = sum(amt * values.get(R.res_ids[r], 0.0) for r, amt in R.cost[ai])
            fin = R.finisher[ai]
            if aura.duration_by_points and fin >= 0:
                top = min(len(aura.duration_by_points) - 1, int(R.res[fin].max))
                ways = [(aura.duration_by_points[p], p * values.get(R.res_ids[fin], 0.0)) for p in range(1, top + 1)]
            else:
                ways = [(aura.duration, 0.0)]
            for dur, extra in ways:
                if dur > a.cooldown:
                    out[idx] = min(out[idx], (base + extra) / dur)
    _upkeep_cache[R] = (buffs, debuffs)
    return buffs, debuffs


def debuff_rate(sim: Sim, di: int, stacks: int) -> float:
    """Damage-equivalent value per second of a debuff with this many stacks on one enemy."""
    per_stack = sum(v * SCHOOL_SHARE.get(s, 0.1) for s, v in sim.R.debuffs[di].damage_taken.items())
    return sim.spec.reference_dps * per_stack * stacks


def debuff_value(sim: Sim, di: int, e: Enemy) -> float:
    d = sim.R.debuffs[di]
    cur = sim.debuff(e, di)
    remaining = cur[0] - sim.t if cur else 0.0
    stacks = cur[1] if cur else 0
    span = min(d.duration, e.time_to_die)
    if stacks < d.max_stacks:
        return debuff_rate(sim, di, 1) * (stacks * max(0.0, span - remaining) + span)
    return debuff_rate(sim, di, d.max_stacks) * max(0.0, span - remaining)


def option_value(sim: Sim, opt: tuple[int, int]) -> float:
    """Net expected value of an option in damage units, using the spec's resource exchange rates."""
    R = sim.R
    values = sim.spec.resource_value
    ai, tid = opt
    if ai == POOL:
        waste = 0.0
        for r in R.ticked:
            res = R.res[r]
            if sim.next_tick[r] - sim.t <= 1.0:
                waste += max(0.0, sim.res[r] + res.tick_amount - res.max) * values.get(res.id, 0.0)
        return -waste
    a = R.abilities[ai]
    e = sim.enemy(tid) if tid >= 0 else sim.enemy(sim.target)
    v, _ = expected_strike(sim, ai, e)
    if a.dot is not None:
        new, lost = expected_dot(sim, ai, e)
        v += new - lost
    if R.debuff_of[ai] >= 0 and e is not None:
        v += debuff_value(sim, R.debuff_of[ai], e)
    fin = R.finisher[ai]
    points = sim.points(fin, e) if fin >= 0 else None
    if R.buff_of[ai] >= 0:
        v += buff_value(sim, R.buff_of[ai], points)
    for r, amt in sim.cost_of(ai):
        v -= amt * values.get(R.res_ids[r], 0.0)
    if fin >= 0:
        v -= (points or 0.0) * values.get(R.res_ids[fin], 0.0)
    for r, amt in R.gain[ai]:
        cur = sim.points(r, e)
        if r in R.on_target and e is not None and sim.res_target[r] not in (-1, e.id):
            v -= sim.res[r] * values.get(R.res_ids[r], 0.0)
        v += min(amt, R.res[r].max - cur) * values.get(R.res_ids[r], 0.0)
    for r, target in R.set[ai]:
        v += (min(target, R.res[r].max) - sim.res[r]) * values.get(R.res_ids[r], 0.0)
    if a.gap_closer and e is not None:
        v += sim.spec.reference_dps * max(0.0, sim.dist(e) - sim.spec.melee_range) / sim.spec.run_speed
    return v
