"""Rollout teacher: scores every legal option by simulating ahead, without seeing the future.

For a decision state the teacher draws K samples of the future. Each sample is a fresh draw of the
scenario's scripted events plus a fresh seed for combat rolls, so the teacher never sees events or
rolls the player couldn't know. Every option is evaluated on every sample (common random numbers):
clone the state, step the option, then follow a rollout policy until the horizon. An option's value
on a sample is the expected damage dealt over the horizon plus what the end state still owes:
banked resources, pending DoT ticks, the rest of active auras, and uses of long cooldowns still to
come (`state_value`). If the fight ends inside the horizon, the time saved is worth reference DPS.

Targets are soft. The softmax temperature for each option comes from the standard error of its
paired difference to the best option, with a floor, so near-ties and noise stay soft.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .sim import EPS, Sim
from .valuation import aura_upkeep, buff_rate, debuff_rate, expected_strike

Option = tuple[int, int]


@dataclass
class TeacherConfig:
    samples: int = 16  # futures per state, shared by every option; 8 is too noisy to rank close calls
    horizon: float = 20.0  # seconds simulated after the decision
    survive_after: float = 12.0  # the survival label asks whether the target lives this much longer
    tau_floor: float = 0.1  # softmax temperature floor, in seconds of reference DPS
    stakes_cutoffs: tuple[float, ...] = (0.25, 1.0, 2.5)  # seconds of reference DPS
    spend_time: float = 3.0  # banked resources count less if the fight has fewer seconds left


@dataclass
class Label:
    """The teacher's verdict on one decision state. `q` and `alive` are the raw rollout results; the
    other fields derive from them and can be recomputed with `summarize`."""

    t: float
    options: list[Option]
    q: np.ndarray  # (options, samples): value of each option on each sampled future, in damage
    alive: np.ndarray  # (options, samples): whether the current target was alive survive_after later
    mean: np.ndarray  # (options,)
    gap: np.ndarray  # (options,) regret against the best option, in damage
    se: np.ndarray  # (options,) standard error of the paired difference to the best option
    soft: np.ndarray  # (options,) soft target, sums to 1
    best: int
    stakes: float  # mean regret over legal options, in seconds of reference DPS
    stakes_level: int
    survive: float  # P(current target is alive survive_after from now)


def act(policy, sims: Sequence[Sim], opts: Sequence[list[Option]]) -> list[Option]:
    """Ask a policy for one decision per sim. Policies with a `batch` method decide all at once."""
    batch = getattr(policy, "batch", None)
    if batch is not None:
        return batch(sims, opts)
    return [policy(s, o) for s, o in zip(sims, opts)]


# ---------------------------------------------------------------------- terminal value


def fight_left(sim: Sim, t_end: float) -> float:
    """Seconds the fight is expected to last: the boss's time to die, or the last enemy's."""
    if sim.has_boss:
        ttd = min((e.time_to_die for e in sim.enemies if e.boss), default=0.0)
    else:
        ttd = max((e.time_to_die for e in sim.enemies), default=0.0)
    return max(0.0, min(ttd, t_end - sim.t))


def _one_use_saving(sim: Sim, bi: int) -> float:
    """What a buff consumed by the next paid ability saves: the best single ability's discounted cost."""
    b = sim.R.buffs[bi]
    values = sim.spec.resource_value
    best = 0.0
    for costs in sim.R.cost:
        saving = 0.0
        for r, amt in costs:
            m = b.cost_mult.get(sim.R.res_ids[r], b.cost_mult.get("all", 1.0))
            saving += amt * (1.0 - m) * values.get(sim.R.res_ids[r], 0.0)
        best = max(best, saving)
    return best


def _buff_owed(sim: Sim, bi: int, left: float) -> float:
    b = sim.R.buffs[bi]
    rem = min(sim.buff_until[bi] - sim.t, left)
    if rem <= 0.0:
        return 0.0
    if b.consumed_by == "cost":
        return _one_use_saving(sim, bi) + buff_rate(sim, bi, costs=False) * min(rem, sim.spec.gcd)
    if b.consumed_by == "swing":
        sp = sim.spec
        swings = (1.0 / sp.main_hand.speed + (1.0 / sp.off_hand.speed if sp.off_hand else 0.0)) * sim.haste()
        rem = min(rem, max(1, sim.buff_charges[bi]) / swings)
    elif b.consumed_by:
        rem = min(rem, 2.0 * sim.spec.gcd)
    # An aura the player keeps up anyway is only worth the upkeep its remaining time saves.
    return min(buff_rate(sim, bi), aura_upkeep(sim)[0][bi]) * rem


def _use_value(sim: Sim, ai: int, avail: float, spend_time: float) -> float:
    """Rough value of one use of an ability with `avail` seconds of fight left: resources gained, its
    buff, and its direct damage on the current target, net of its cost. Never negative (it's optional)."""
    R = sim.R
    values = sim.spec.resource_value
    v = sum(amt * values.get(R.res_ids[r], 0.0) for r, amt in R.gain[ai] if r not in R.on_target)
    v -= sum(amt * values.get(R.res_ids[r], 0.0) for r, amt in R.cost[ai])
    bi = R.buff_of[ai]
    if bi >= 0:
        b = R.buffs[bi]
        dur = max([b.duration, *b.duration_by_points])
        v += min(buff_rate(sim, bi), aura_upkeep(sim)[0][bi]) * min(dur, avail)
    if R.abilities[ai].damage is not None:
        v += expected_strike(sim, ai, sim.enemy(sim.target))[0]
    return max(0.0, v) * min(1.0, avail / spend_time)


def _cooldowns_owed(sim: Sim, left: float, horizon: float, spend_time: float) -> float:
    """Uses of long cooldowns still possible before the fight ends, at their use values. A rollout can't
    see a cooldown that won't come back within its horizon, so the end state has to carry it."""
    R, t = sim.R, sim.t
    v = 0.0
    for ai, a in enumerate(R.abilities):
        if a.cooldown <= horizon:
            continue
        g = R.group[ai]
        ready = max(0.0, sim.cd_until[ai] - t, sim.group_until[g] - t if g >= 0 else 0.0)
        while ready < left:
            v += _use_value(sim, ai, left - ready, spend_time)
            ready += a.cooldown
    return v


def state_value(sim: Sim, t_end: float, spend_time: float = 3.0, horizon: float = 20.0) -> float:
    """Damage the state still owes at the spec's exchange rates: banked resources, DoT ticks that land
    before their target dies, the rest of active buffs and debuffs, and uses of cooldowns longer than
    the rollout horizon. t_end is the fight's hard cap."""
    left = fight_left(sim, t_end)
    if left <= 0.0:
        return 0.0
    R, t = sim.R, sim.t
    values = sim.spec.resource_value
    upkeep = aura_upkeep(sim)[1]
    v = 0.0
    for r in range(R.n_res):
        amt, val = sim.res[r], values.get(R.res_ids[r], 0.0)
        if amt <= 0.0 or val == 0.0:
            continue
        usable = left
        if r in R.on_target:
            holder = sim.enemy(sim.res_target[r])
            if holder is None:
                continue
            usable = min(usable, holder.time_to_die)
        v += val * amt * min(1.0, usable / spend_time)
    for e in sim.enemies:
        life = min(e.time_to_die, left)
        owed = 0.0
        for nt, ticks, dmg, iv, school in e.dots.values():
            if nt - t <= life:
                owed += min(ticks, int((life - (nt - t)) / iv) + 1) * dmg * sim.damage_mult(e, school)
        v += min(owed, e.hp)
        for di, (until, stacks, _) in e.debuffs.items():
            if until > t:
                v += min(debuff_rate(sim, di, stacks), upkeep[di]) * min(until - t, life)
    for bi in range(len(R.buffs)):
        if sim.buff_until[bi] > t:
            v += _buff_owed(sim, bi, left)
    return v + _cooldowns_owed(sim, left, horizon, spend_time)


# ---------------------------------------------------------------------- rollouts


def rollouts(
    sim: Sim,
    opts: list[Option],
    seeds: list[int],
    futures: list[list],
    policy,
    horizon: float,
    survive_after: float,
) -> tuple[list[Sim], list[bool]]:
    """Run every option on every sampled future, stepping all branches in lockstep so a batched policy
    can decide for all of them at once. Returns the finished branches (option-major) and whether the
    current target was alive survive_after seconds from now in each."""
    t_end = min(sim.t_max, sim.t + horizon)
    mark = sim.t + survive_after
    tid = sim.target
    runs: list[Sim] = []
    for o in opts:
        for seed, events in zip(seeds, futures):
            c = sim.clone(seed)
            c.events, c.ev_i = events, 0
            c.t_max = t_end
            c.step(o)
            runs.append(c)
    alive: list[bool | None] = [None] * len(runs)
    live = list(range(len(runs)))
    while live:
        nxt = []
        for i in live:
            c = runs[i]
            if alive[i] is None and (c.done or c.t >= mark - EPS):
                alive[i] = c.enemy(tid) is not None
            if not c.done:
                nxt.append(i)
        live = nxt
        if live:
            sims = [runs[i] for i in live]
            for c, o in zip(sims, act(policy, sims, [c.legal_options() for c in sims])):
                c.step(o)
    return runs, [bool(a) for a in alive]


def summarize(q: np.ndarray, ref_dps: float, cfg: TeacherConfig) -> dict:
    """Soft target, regret, and stakes from a (options, samples) matrix of rollout values."""
    n, k = q.shape
    mean = q.mean(axis=1)
    best = int(mean.argmax())
    diff = q - q[best]
    gap = -diff.mean(axis=1)
    se = diff.std(axis=1, ddof=1) / math.sqrt(k) if k > 1 else np.zeros(n)
    tau = np.sqrt(se**2 + (cfg.tau_floor * ref_dps) ** 2)
    logits = -gap / tau
    soft = np.exp(logits - logits.max())
    soft /= soft.sum()
    stakes = float(gap.mean() / ref_dps)
    level = int(np.searchsorted(np.asarray(cfg.stakes_cutoffs), stakes, side="right"))
    return dict(mean=mean, gap=gap, se=se, soft=soft, best=best, stakes=stakes, stakes_level=level)


class Teacher:
    def __init__(self, policy, cfg: TeacherConfig | None = None):
        self.policy = policy
        self.cfg = cfg or TeacherConfig()

    def future(self, sim: Sim, rng: random.Random, t_end: float) -> list:
        """One draw of the scripted events between now and t_end, from what the player can know."""
        return sim.plan.sample(rng, sim.t, t_end, sim.last_event)

    def evaluate(self, sim: Sim, rng: random.Random, opts: list[Option] | None = None) -> Label:
        cfg = self.cfg
        opts = sim.legal_options() if opts is None else opts
        t_end = min(sim.t_max, sim.t + cfg.horizon)
        seeds = [rng.getrandbits(62) for _ in range(cfg.samples)]
        futures = [self.future(sim, random.Random(rng.getrandbits(62)), t_end) for _ in range(cfg.samples)]
        runs, alive = rollouts(sim, opts, seeds, futures, self.policy, cfg.horizon, cfg.survive_after)
        ref = sim.spec.reference_dps
        q = np.empty(len(runs))
        for i, c in enumerate(runs):
            gained = c.ev_damage - sim.ev_damage
            if c.t >= t_end - EPS:
                q[i] = gained + state_value(c, sim.t_max, cfg.spend_time, cfg.horizon)
            else:  # the fight ended early: the time saved is worth reference DPS
                q[i] = gained + ref * (t_end - c.t)
        q = q.reshape(len(opts), cfg.samples)
        alive_m = np.asarray(alive, dtype=bool).reshape(len(opts), cfg.samples)
        s = summarize(q, ref, cfg)
        survive = min(1.0, float((s["soft"][:, None] * alive_m).sum() / cfg.samples))
        return Label(t=sim.t, options=list(opts), q=q, alive=alive_m, survive=survive, **s)


class TeacherPolicy:
    """Plays the teacher's best option at every decision with a real choice. Slow; for evaluation."""

    name = "teacher"

    def __init__(self, teacher: Teacher, rng: random.Random | None = None):
        self.teacher = teacher
        self.rng = rng or random.Random(0)

    def __call__(self, sim: Sim, opts: list[Option]) -> Option:
        if len(opts) == 1:
            return opts[0]
        label = self.teacher.evaluate(sim, self.rng, opts)
        return label.options[label.best]

