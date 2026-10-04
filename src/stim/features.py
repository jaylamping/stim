"""Generic entity tokens for the decision model.

A decision state becomes a set of tokens, one array per type: the player, each resource, each aura
(buffs, and debuffs as seen on the current target), each ability (legal or not), each enemy, and each
legal option. Abilities are described by what they do (costs by resource kind, damage, DoTs, auras,
requirements), never by name or index, so a model trained on several specs can read a new one.
Option tokens point at their ability and enemy tokens.

Units: damage in seconds of the spec's reference DPS, time in units of 10 s, resources as a fraction
of their pool. Positions use a frame centred on the player with x pointing at the current target, so
the features don't change when the whole fight is rotated.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import numpy as np

from .sim import EPS, POOL, Sim
from .valuation import (aura_upkeep, buff_rate, buff_value, debuff_rate, debuff_value, expected_dot,
                        expected_strike, option_value)

VERSION = 1  # bump when token layouts change; caches and checkpoints carry it
RES_KINDS = ("energy", "rage", "mana", "secondary")
EVENT_KINDS = ("knockback", "turn", "reposition", "spawn")
CONSUMED = ("", "cost", "swing", "ability", "school")
MAX_ENEMIES = 12
T = 10.0  # time unit


def _schools(d: dict[str, float]) -> list[float]:
    """physical, bleed, everything else"""
    return [d.get("physical", 0.0), d.get("bleed", 0.0), sum(v for k, v in d.items() if k not in ("physical", "bleed"))]


def _school_onehot(school: str) -> list[float]:
    return [float(school == "physical"), float(school == "bleed"), float(school not in ("physical", "bleed"))]


def _kind_vec(sim: Sim, amounts: list[tuple[int, float]], scale: bool = True) -> list[float]:
    """Per-resource amounts folded into the four resource kinds, as a fraction of the pool."""
    out = [0.0] * 4
    for r, amt in amounts:
        res = sim.R.res[r]
        out[RES_KINDS.index(res.kind)] += amt / res.max if scale else amt
    return out


# ---------------------------------------------------------------------- static descriptors

AURA_STATIC = 25


def _aura_static(sim: Sim, aura, debuff: bool) -> list[float]:
    R = sim.R
    discount = [0.0] * 4
    for k, m in aura.cost_mult.items():
        kinds = RES_KINDS if k == "all" else (R.res[R.res_index[k]].kind,)
        for kind in kinds:
            discount[RES_KINDS.index(kind)] = 1.0 - m
    regen = [0.0] * 4
    for k, m in aura.regen_mult.items():
        regen[RES_KINDS.index(R.res[R.res_index[k]].kind)] = m - 1.0
    dur = max([aura.duration, *aura.duration_by_points])
    consumed = aura.consumed_by.split(":")[0]
    if consumed not in CONSUMED:
        consumed = "school"
    return [
        float(debuff), aura.haste, aura.damage_mult, aura.crit_bonus, *discount, *regen,
        aura.cleave_targets / 3.0, *_schools(aura.damage_taken), dur / T, float(bool(aura.duration_by_points)),
        aura.max_stacks / 5.0, aura.charges / 5.0, *[float(consumed == c) for c in CONSUMED],
    ]


ABILITY_STATIC = 69

_static_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _ability_static(sim: Sim) -> np.ndarray:
    R = sim.R
    cached = _static_cache.get(R)
    if cached is not None:
        return cached
    sp = sim.spec
    ref = sp.reference_dps
    values = sp.resource_value
    rows = []
    proc_abilities = {p.trigger[8:] for p in sp.procs if p.trigger.startswith("ability:")}
    for ai, a in enumerate(R.abilities):
        fin = R.finisher[ai]
        top = R.res[fin].max if fin >= 0 else 0.0
        cost_dmg = sum(amt * values.get(R.res_ids[r], 0.0) for r, amt in R.cost[ai]) / ref
        dmg = a.damage
        base = (dmg.weapon * sp.main_hand.damage + dmg.flat) / ref if dmg else 0.0
        by_pts = (max(dmg.by_points) / ref) if dmg and dmg.by_points else 0.0
        dot = a.dot
        dot_total = (dot.ticks * (dot.per_tick + dot.per_tick_per_point * top) / ref) if dot else 0.0
        extra = R.extra[ai]
        buff = _aura_static(sim, R.buffs[R.buff_of[ai]], False) if R.buff_of[ai] >= 0 else [0.0] * AURA_STATIC
        debuff = _aura_static(sim, R.debuffs[R.debuff_of[ai]], True) if R.debuff_of[ai] >= 0 else [0.0] * AURA_STATIC
        gains_on_target = any(r in R.on_target for r, _ in R.gain[ai])
        row = [
            *_kind_vec(sim, R.cost[ai]), cost_dmg,
            *_kind_vec(sim, R.gain[ai]), float(gains_on_target),
            *_kind_vec(sim, R.set[ai]), float(bool(R.set[ai])),
            math.log1p(a.cooldown / T), R.gcd[ai], float(a.off_gcd), a.cast_time, float(a.next_swing),
            float(fin >= 0),
            float(dmg is not None), base, by_pts, *(_school_onehot(dmg.school) if dmg else [0.0] * 3),
            float(dmg.can_crit) if dmg else 0.0, dmg.crit_bonus if dmg else 0.0,
            (dmg.aoe_radius / T) if dmg else 0.0, (dmg.max_targets / 10.0) if dmg else 0.0,
            (dmg.extra_targets / 3.0) if dmg else 0.0,
            float(dot is not None), dot_total, (dot.ticks * dot.interval / T) if dot else 0.0,
            float(dot is not None and dot.per_tick_per_point > 0),
            float(extra is not None), (extra[1] / R.res[extra[0]].max) if extra else 0.0,
            (extra[1] * extra[2] / ref) if extra else 0.0,
            float(R.buff_of[ai] >= 0), float(R.debuff_of[ai] >= 0), float(R.consumes[ai] >= 0),
            float(a.gap_closer), a.range_min / T, a.range_max / T,
            float(a.req_behind), a.req_target_health_below, a.req_target_health_above,
            float(R.req_buff[ai] >= 0), float(R.req_no_buff[ai] >= 0), float(a.id in proc_abilities),
            float(R.targeted[ai]),
        ]
        # what the applied buff does (haste, damage, crit, discounts, regen, cleave, duration) and the
        # applied debuff (damage taken by school, duration)
        rows.append(row + buff[1:13] + [buff[16]] + debuff[13:17])
    arr = np.asarray(rows, dtype=np.float32).reshape(len(R.abilities), -1)
    assert arr.shape[1] == ABILITY_STATIC, arr.shape
    _static_cache[R] = arr
    return arr


# ---------------------------------------------------------------------- per-state tokens

PLAYER = 31
RESOURCE = 17
AURA = AURA_STATIC + 7
ABILITY = ABILITY_STATIC + 5
ENEMY = 27
OPTION = 34


@dataclass
class Tokens:
    player: np.ndarray  # (PLAYER,)
    res: np.ndarray  # (resources, RESOURCE)
    aura: np.ndarray  # (auras, AURA)
    abil: np.ndarray  # (abilities, ABILITY)
    enemy: np.ndarray  # (enemies, ENEMY)
    opt: np.ndarray  # (options, OPTION)
    opt_abil: np.ndarray  # (options,) ability token index, -1 for pooling
    opt_enemy: np.ndarray  # (options,) enemy token index, -1 for none


def _option_row(is_pool=0.0, wait=0.0, strike=0.0, hits=0.0, new=0.0, lost=0.0, deb=0.0, bval=0.0, cost_dmg=0.0,
                resets=0.0, waste=0.0, value=0.0, cost_k=(0.0,) * 4, after_k=(0.0,) * 4, pts=0.0, is_tgt=0.0, d=0.0,
                behind=0.0, dot_rem=0.0, dot_ticks=0.0, deb_rem=0.0, deb_stacks=0.0, off_gcd=0.0, next_swing=0.0,
                gcd=0.0, targeted=0.0, hp=0.0, ttd=0.0) -> list[float]:
    return [is_pool, wait, strike, hits, new, lost, deb, bval, cost_dmg, resets, waste, value, *cost_k, *after_k, pts,
            is_tgt, d, behind, dot_rem, dot_ticks, deb_rem, deb_stacks, off_gcd, next_swing, gcd, targeted, hp, ttd]


def _frame(sim: Sim) -> tuple[float, float]:
    tgt = sim.enemy(sim.target)
    if tgt is not None:
        dx, dy = tgt.x - sim.px, tgt.y - sim.py
        n = math.hypot(dx, dy)
        if n > 1e-6:
            return dx / n, dy / n
    return 1.0, 0.0


def tokens(sim: Sim, opts: list[tuple[int, int]] | None = None) -> Tokens:
    R, sp, t = sim.R, sim.spec, sim.t
    ref = sp.reference_dps
    opts = sim.legal_options() if opts is None else opts
    tgt = sim.enemy(sim.target)

    # enemies: the target first, then by distance; at most MAX_ENEMIES
    dists = {e.id: sim.dist(e) for e in sim.enemies}
    order = sorted(sim.enemies, key=lambda e: (e.id != sim.target, dists[e.id]))[:MAX_ENEMIES]
    slot = {e.id: i for i, e in enumerate(order)}
    ux, uy = _frame(sim)
    upkeep_b, upkeep_d = aura_upkeep(sim)
    in_melee = sum(1 for d in dists.values() if d <= sp.melee_range)
    near = sum(1 for d in dists.values() if d <= 8.0)
    enemies = []
    for e in order:
        dx, dy = e.x - sim.px, e.y - sim.py
        fx, fy = sim.facing(e)
        d = dists[e.id]
        dots = [(nt - t + (left - 1) * iv, left * dmg) for nt, left, dmg, iv, _ in e.dots.values()]
        bonus = [0.0, 0.0, 0.0]
        deb_rem = 0.0
        for di, (until, stacks, _) in e.debuffs.items():
            if until > t:
                for i, v in enumerate(_schools(R.debuffs[di].damage_taken)):
                    bonus[i] += v * stacks
                deb_rem = max(deb_rem, until - t)
        held = [sim.res[r] / R.res[r].max for r in R.on_target if sim.res_target[r] == e.id]
        to_player = (dx * fx + dy * fy) / d if d > 1e-6 else 0.0
        enemies.append([
            (dx * ux + dy * uy) / T, (-dx * uy + dy * ux) / T, d / T, fx * ux + fy * uy, -fx * uy + fy * ux,
            -to_player, float(d <= sp.melee_range), float(sim.behind(e)),
            float(e.id == sim.target), float(e.boss), float(e.engaged), float(e.turn_until > t),
            max(0.0, e.turn_until - t) / T,
            e.hp / e.hp_max, math.log10(max(e.hp, 1.0)) / 6.0, e.hp / ref / 100.0, e.loss_rate / ref,
            min(e.time_to_die, 300.0) / 100.0,
            len(dots) / 3.0, sum(dmg for _, dmg in dots) / ref, max((r for r, _ in dots), default=0.0) / T,
            *bonus, deb_rem / T, float(bool(held)), held[0] if held else 0.0,
        ])

    # player
    since = [min(60.0, t - sim.last_event.get(k, 0.0)) / 60.0 for k in EVENT_KINDS]
    oh = sp.off_hand
    player = [
        max(0.0, sim.gcd_until - t), sp.gcd, max(0.0, sim.swing[0] - t), sp.main_hand.speed,
        (max(0.0, sim.swing[1] - t) if oh else 0.0), (oh.speed if oh else 0.0), float(oh is not None),
        sim.haste() - 1.0, sim.crit_chance(), sp.crit_multiplier - 1.0, sp.miss_chance + sp.dodge_chance,
        sp.glancing_chance, sp.dual_wield_miss if oh else 0.0,
        max(0.0, sim.knocked_until - t), float(sim.casting >= 0), max(0.0, sim.cast_until - t) if sim.casting >= 0 else 0.0,
        float(sim.queued >= 0), t / 100.0, len(sim.enemies) / 5.0, in_melee / 5.0, near / 5.0,
        float(sim.has_boss), (dists[tgt.id] / T) if tgt else 0.0, float(tgt is None),
        sp.main_hand.damage / sp.main_hand.speed / ref, (oh.damage / oh.speed / ref) if oh else 0.0,
        *since, sp.melee_range / T,
    ]

    # resources
    res = []
    for r, rs in enumerate(R.res):
        regen_m = sim._regen_mult(r)
        cost_m = 1.0
        for bi, mults in R.cost_mult_buffs:
            if sim.buff_until[bi] > t:
                for k, m in mults:
                    if k == -1 or k == r:
                        cost_m *= m
        holder = sim.enemy(sim.res_target[r]) if r in R.on_target else None
        res.append([
            *[float(rs.kind == k) for k in RES_KINDS], sim.res[r] / rs.max, sim.points(r, tgt) / rs.max,
            math.log10(rs.max) / 4.0, rs.regen_per_sec / rs.max, rs.tick_amount / rs.max, rs.tick_interval,
            (sim.next_tick[r] - t) if rs.tick_interval > 0 else 0.0, rs.per_white_damage * ref / rs.max,
            float(r in R.on_target), float(holder is not None and holder is tgt),
            sp.resource_value.get(rs.id, 0.0) * rs.max / ref, regen_m - 1.0, 1.0 - cost_m,
        ])

    # auras: buffs, then debuffs as they stand on the current target
    auras = []
    for bi, b in enumerate(R.buffs):
        rem = max(0.0, sim.buff_until[bi] - t)
        dur = max([b.duration, *b.duration_by_points]) or 1.0
        auras.append(_aura_static(sim, b, False) + [
            float(rem > 0), rem / T, rem / dur, (sim.buff_stacks[bi] / b.max_stacks) if rem > 0 else 0.0,
            (sim.buff_charges[bi] / max(1, b.charges)) if rem > 0 and b.charges else 0.0,
            buff_rate(sim, bi) / ref, min(upkeep_b[bi], 10.0 * ref) / ref,
        ])
    for di, d in enumerate(R.debuffs):
        cur = sim.debuff(tgt, di) if tgt is not None else None
        rem = cur[0] - t if cur else 0.0
        dur = d.duration or 1.0
        auras.append(_aura_static(sim, d, True) + [
            float(cur is not None), rem / T, rem / dur, (cur[1] / d.max_stacks) if cur else 0.0,
            (cur[2] / max(1, d.charges)) if cur and d.charges else 0.0,
            debuff_rate(sim, di, 1) / ref, min(upkeep_d[di], 10.0 * ref) / ref,
        ])

    # abilities
    static = _ability_static(sim)
    legal = {}
    for o in opts:
        if o[0] >= 0:
            legal[o[0]] = legal.get(o[0], 0) + 1
    dyn = []
    for ai in range(R.n_abilities):
        g = R.group[ai]
        cd = max(0.0, sim.cd_until[ai] - t, (sim.group_until[g] - t) if g >= 0 else 0.0)
        dyn.append([math.log1p(cd / T), float(cd <= EPS), float(sim.affordable(ai)), float(ai in legal), legal.get(ai, 0) / 5.0])
    abil = np.concatenate([static, np.asarray(dyn, dtype=np.float32).reshape(-1, 5)], axis=1)

    # options
    values = sp.resource_value
    now_kind = [0.0] * 4
    for r, rs in enumerate(R.res):
        now_kind[RES_KINDS.index(rs.kind)] = sim.res[r] / rs.max
    opt_rows, opt_abil, opt_enemy = [], [], []
    for ai, tid in opts:
        if ai == POOL:
            wait = min(sim.swing[0], sim.swing[1], min(sim.next_tick), t + 1.0) - t
            opt_rows.append(_option_row(is_pool=1.0, wait=max(0.05, wait), value=option_value(sim, (ai, tid)) / ref,
                                        after_k=now_kind))
            opt_abil.append(-1)
            opt_enemy.append(-1)
            continue
        e = sim.enemy(tid) if tid >= 0 else None
        strike, hits = expected_strike(sim, ai, e if e is not None else tgt)
        new, lost = expected_dot(sim, ai, e)
        fin = R.finisher[ai]
        pts = sim.points(fin, e) if fin >= 0 else 0.0
        cost = sim.cost_of(ai)
        after = list(sim.res)
        for r, amt in cost:
            after[r] = max(0.0, after[r] - amt)
        waste = resets = 0.0
        for r, amt in R.gain[ai]:
            if r in R.on_target and e is not None and sim.res_target[r] not in (-1, e.id):
                resets += sim.res[r] / R.res[r].max
                after[r] = 0.0
            waste += max(0.0, amt - (R.res[r].max - after[r])) / R.res[r].max
            after[r] = min(R.res[r].max, after[r] + amt)
        for r, v in R.set[ai]:
            after[r] = min(R.res[r].max, v)
        if fin >= 0:
            after[fin] = 0.0
        after_kind = [0.0] * 4
        for r, rs in enumerate(R.res):
            after_kind[RES_KINDS.index(rs.kind)] = after[r] / rs.max
        cur_dot = e.dots.get(ai) if e is not None else None
        cur_deb = sim.debuff(e, R.debuff_of[ai]) if e is not None and R.debuff_of[ai] >= 0 else None
        opt_rows.append(_option_row(
            strike=strike / ref, hits=hits / 5.0, new=new / ref, lost=lost / ref,
            deb=(debuff_value(sim, R.debuff_of[ai], e) / ref) if e is not None and R.debuff_of[ai] >= 0 else 0.0,
            bval=(buff_value(sim, R.buff_of[ai], pts if fin >= 0 else None) / ref) if R.buff_of[ai] >= 0 else 0.0,
            cost_dmg=sum(amt * values.get(R.res_ids[r], 0.0) for r, amt in cost) / ref, resets=resets, waste=waste,
            value=option_value(sim, (ai, tid)) / ref, cost_k=_kind_vec(sim, cost), after_k=after_kind,
            pts=(pts / R.res[fin].max) if fin >= 0 else 0.0, is_tgt=float(e is not None and e is tgt),
            d=(sim.dist(e) / T) if e is not None else 0.0, behind=float(e is not None and sim.behind(e)),
            dot_rem=((cur_dot[0] - t + (cur_dot[1] - 1) * cur_dot[3]) / T) if cur_dot else 0.0,
            dot_ticks=(cur_dot[1] / 10.0) if cur_dot else 0.0,
            deb_rem=((cur_deb[0] - t) / T) if cur_deb else 0.0, deb_stacks=(cur_deb[1] / 5.0) if cur_deb else 0.0,
            off_gcd=float(R.gcd[ai] == 0.0), next_swing=float(R.abilities[ai].next_swing), gcd=R.gcd[ai],
            targeted=float(R.targeted[ai]), hp=(e.hp / e.hp_max) if e is not None else 0.0,
            ttd=(min(e.time_to_die, 300.0) / 100.0) if e is not None else 0.0,
        ))
        opt_abil.append(ai)
        opt_enemy.append(slot.get(tid, -1) if tid >= 0 else -1)

    f32 = np.float32
    out = Tokens(
        player=np.asarray(player, dtype=f32),
        res=np.asarray(res, dtype=f32).reshape(-1, RESOURCE),
        aura=np.asarray(auras, dtype=f32).reshape(-1, AURA),
        abil=abil.astype(f32),
        enemy=np.asarray(enemies, dtype=f32).reshape(-1, ENEMY),
        opt=np.asarray(opt_rows, dtype=f32).reshape(-1, OPTION),
        opt_abil=np.asarray(opt_abil, dtype=np.int64),
        opt_enemy=np.asarray(opt_enemy, dtype=np.int64),
    )
    assert out.player.shape == (PLAYER,), out.player.shape
    return out
