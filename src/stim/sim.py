"""Event-driven combat simulator for melee specs.

The sim only encodes generic game rules (resources, swings, procs, auras, DoTs, positioning). It
has no opinion about what to press: each decision point is handed to a policy as a list of legal
options, each an (ability index, target enemy id) pair. Class behavior comes entirely from the spec.
"""

from __future__ import annotations

import math
import random

from .spec import Ability, Aura, Damage, Spec

POOL = -1  # ability index of the "wait / pool resources" option
EPS = 1e-9
INF = float("inf")
MOVE_STEP = 0.25  # max integration step while anything is moving
TANK_GAP = 2.0  # tanked enemies stand this far from the tank
BEHIND_GAP = 2.5  # the player stands this far behind its target
LOSS_TAU = 3.0  # time constant of the observed health-loss rate
KNOCKED_TIME = 0.6  # a knockback stops the player from acting or moving for this long
MAX_PROC_DEPTH = 2
MH, OH = 0, 1
ALL = -1  # cost_mult key meaning "every resource"
# Random streams. Swings of each hand use streams MH and OH; each ability use draws one block from
# ABILITY_STREAM; extra targets of a strike use SPLASH_STREAM; proc strikes and extra attacks use
# NESTED_STREAM. Every attack draws a fixed-size block (hit roll, crit roll, one roll per proc), so
# branches cloned from one state with the same seed get aligned luck: the n-th main-hand swing or the
# n-th ability use sees the same numbers whatever happened in between.
ABILITY_STREAM, SPLASH_STREAM, NESTED_STREAM = 2, 3, 4
N_STREAMS = 5


class Rules:
    """A spec compiled into index-based lookups for the hot loop."""

    def __init__(self, spec: Spec):
        self.spec = spec
        self.res_ids = [r.id for r in spec.resources]
        ri = {r: i for i, r in enumerate(self.res_ids)}
        self.res_index = ri
        self.res = spec.resources
        self.n_res = len(spec.resources)
        self.ticked = [i for i, r in enumerate(spec.resources) if r.tick_interval > 0]
        self.flowing = [i for i, r in enumerate(spec.resources) if r.regen_per_sec > 0]
        self.white_gain = [(i, r.per_white_damage) for i, r in enumerate(spec.resources) if r.per_white_damage > 0]

        self.buffs = spec.buffs
        self.debuffs = spec.debuffs
        bi = {b.id: i for i, b in enumerate(spec.buffs)}
        di = {d.id: i for i, d in enumerate(spec.debuffs)}
        self.buff_index, self.debuff_index = bi, di
        self.abilities = spec.abilities
        self.n_abilities = len(spec.abilities)
        ai = {a.id: i for i, a in enumerate(spec.abilities)}
        self.ability_index = ai

        def cm(m: dict[str, float]) -> list[tuple[int, float]]:
            return [(ALL if k == "all" else ri[k], v) for k, v in m.items()]

        self.cost_mult_buffs = [(i, cm(b.cost_mult)) for i, b in enumerate(spec.buffs) if b.cost_mult]
        self.regen_mult_buffs = [(i, [(ri[k], v) for k, v in b.regen_mult.items()]) for i, b in enumerate(spec.buffs) if b.regen_mult]
        self.haste_buffs = [(i, b.haste) for i, b in enumerate(spec.buffs) if b.haste]
        self.dmg_buffs = [(i, b.damage_mult) for i, b in enumerate(spec.buffs) if b.damage_mult]
        self.crit_buffs = [(i, b.crit_bonus) for i, b in enumerate(spec.buffs) if b.crit_bonus]
        self.cleave_buffs = [(i, b.cleave_targets, b.cleave_radius) for i, b in enumerate(spec.buffs) if b.cleave_targets]
        self.cost_consumed = [i for i, b in enumerate(spec.buffs) if b.consumed_by == "cost"]
        self.swing_consumed = [i for i, b in enumerate(spec.buffs) if b.consumed_by == "swing"]
        self.ability_consumed: list[list[int]] = [[] for _ in spec.abilities]
        for i, b in enumerate(spec.buffs):
            if b.consumed_by.startswith("ability:"):
                self.ability_consumed[ai[b.consumed_by[8:]]].append(i)
        self.school_consumed: dict[str, list[int]] = {}
        for i, d in enumerate(spec.debuffs):
            if d.consumed_by and not d.consumed_by.startswith("ability:") and d.consumed_by not in ("cost", "swing"):
                self.school_consumed.setdefault(d.consumed_by, []).append(i)

        groups = sorted({a.cooldown_group for a in spec.abilities if a.cooldown_group})
        self.group_index = {g: i for i, g in enumerate(groups)}
        self.n_groups = len(groups)

        self.cost = [[(ri[k], v) for k, v in a.cost.items()] for a in spec.abilities]
        self.gain = [[(ri[k], v) for k, v in a.gain.items()] for a in spec.abilities]
        self.set = [[(ri[k], v) for k, v in a.set.items()] for a in spec.abilities]
        self.finisher = [ri[a.finisher] if a.finisher else -1 for a in spec.abilities]
        self.extra = [(ri[a.extra.resource], a.extra.max, a.extra.damage_per) if a.extra else None for a in spec.abilities]
        self.buff_of = [bi[a.buff] if a.buff else -1 for a in spec.abilities]
        self.debuff_of = [di[a.debuff] if a.debuff else -1 for a in spec.abilities]
        self.consumes = [bi[a.consumes_buff] if a.consumes_buff else -1 for a in spec.abilities]
        self.req_buff = [bi[a.req_buff] if a.req_buff else -1 for a in spec.abilities]
        self.req_no_buff = [bi[a.req_no_buff] if a.req_no_buff else -1 for a in spec.abilities]
        self.group = [self.group_index[a.cooldown_group] if a.cooldown_group else -1 for a in spec.abilities]
        self.gcd = [0.0 if a.off_gcd or a.next_swing else (a.gcd if a.gcd is not None else spec.gcd) for a in spec.abilities]
        self.has_cost = [any(v > 0 for v in a.cost.values()) for a in spec.abilities]
        on_target = {i for i, r in enumerate(spec.resources) if r.on_target}
        self.on_target = on_target
        self.builds_on_target = [any(r in on_target for r, _ in g) for g in self.gain]
        self.targeted = [self._targeted(a, i) for i, a in enumerate(spec.abilities)]

        self.procs = spec.procs
        self.n_rolls = 2 + len(spec.procs)
        self.proc_gain = [[(ri[k], v) for k, v in p.gain.items()] for p in spec.procs]
        self.proc_buff = [bi[p.buff] if p.buff else -1 for p in spec.procs]
        self.proc_debuff = [di[p.debuff] if p.debuff else -1 for p in spec.procs]
        self.proc_req = [bi[p.requires_buff] if p.requires_buff else -1 for p in spec.procs]
        self.by_trigger: dict[str, list[int]] = {}
        for i, p in enumerate(spec.procs):
            self.by_trigger.setdefault(p.trigger, []).append(i)

    def _targeted(self, a: Ability, i: int) -> bool:
        if a.aoe or a.next_swing:
            return False
        fin = self.finisher[i]
        return bool(
            a.damage or a.dot or a.debuff or a.gap_closer or a.req_behind or a.req_target_health_below
            or a.req_target_health_above or self.builds_on_target[i] or (fin >= 0 and fin in self.on_target)
        )


def rules_for(spec: Spec) -> Rules:
    r = getattr(spec, "_rules", None)
    if r is None:
        r = Rules(spec)
        spec._rules = r  # type: ignore[attr-defined]
    return r


class Enemy:
    __slots__ = (
        "id", "name", "boss", "x", "y", "fx", "fy", "turn_until", "speed",
        "hp", "hp_max", "party_dps", "loss_rate", "pending", "dots", "debuffs", "engaged",
    )

    def __init__(self, id: int, name: str, boss: bool, x: float, y: float, hp: float, party_dps: float):
        self.id, self.name, self.boss = id, name, boss
        self.x, self.y = x, y
        self.fx, self.fy = 0.0, -1.0
        self.turn_until = -1.0
        self.speed = 7.0
        self.hp = self.hp_max = hp
        self.party_dps = party_dps
        self.loss_rate = party_dps
        self.pending = 0.0
        # dot key -> (next tick, ticks left, damage per tick, interval, school)
        self.dots: dict[int, tuple[float, int, float, float, str]] = {}
        # debuff index -> (expiry, stacks, charges)
        self.debuffs: dict[int, tuple[float, int, int]] = {}
        self.engaged = False

    def clone(self) -> Enemy:
        e = Enemy.__new__(Enemy)
        e.id, e.name, e.boss = self.id, self.name, self.boss
        e.x, e.y, e.fx, e.fy = self.x, self.y, self.fx, self.fy
        e.turn_until, e.speed = self.turn_until, self.speed
        e.hp, e.hp_max, e.party_dps = self.hp, self.hp_max, self.party_dps
        e.loss_rate, e.pending = self.loss_rate, self.pending
        e.dots, e.debuffs = dict(self.dots), dict(self.debuffs)
        e.engaged = self.engaged
        return e

    @property
    def time_to_die(self) -> float:
        return self.hp / self.loss_rate if self.loss_rate > 1.0 else 999.0


class EventPlan:
    """Rates of a scenario's scripted disruptions. Rollouts resample the future from these, so the
    teacher never sees events the player couldn't have known about."""

    def __init__(
        self,
        knockback: tuple[float, float] | None = None,
        turn: tuple[float, float] | None = None,
        turn_duration: tuple[float, float] = (1.5, 3.0),
        reposition: tuple[float, float] | None = None,
        reposition_distance: tuple[float, float] = (8.0, 16.0),
        wave: tuple[float, float] | None = None,
        wave_size: tuple[int, int] = (2, 3),
        add_hp: tuple[float, float] = (15000.0, 30000.0),
        add_party_dps: tuple[float, float] = (900.0, 1600.0),
    ):
        self.knockback, self.turn, self.turn_duration = knockback, turn, turn_duration
        self.reposition, self.reposition_distance = reposition, reposition_distance
        self.wave, self.wave_size, self.add_hp, self.add_party_dps = wave, wave_size, add_hp, add_party_dps

    def sample(
        self, rng: random.Random, t0: float, t_max: float, last: dict[str, float] | None = None
    ) -> list[tuple[float, str, tuple]]:
        """Events in [t0, t_max). Each kind recurs at uniformly random intervals; `last` holds when each
        kind last happened (default: the pull at t=0), and the first draw is conditioned on the time
        already elapsed since then, which a player can see."""
        events: list[tuple[float, str, tuple]] = []
        last = last or {}

        def process(every: tuple[float, float] | None, kind: str, payload) -> None:
            if not every:
                return
            lo, hi = every
            prev = last.get(kind, 0.0)
            since = t0 - prev
            t = prev + rng.uniform(max(lo, since), hi) if since < hi else t0
            while t < t_max:
                events.append((t, kind, payload()))
                t += rng.uniform(lo, hi)

        process(self.knockback, "knockback", lambda: (rng.uniform(12.0, 18.0),))
        process(self.turn, "turn", lambda: (rng.uniform(*self.turn_duration),))
        process(self.reposition, "reposition", lambda: (rng.uniform(0, 2 * math.pi), rng.uniform(*self.reposition_distance)))
        process(
            self.wave,
            "spawn",
            lambda: tuple(
                (rng.uniform(*self.add_hp), rng.uniform(*self.add_party_dps), rng.uniform(0, 2 * math.pi))
                for _ in range(rng.randint(*self.wave_size))
            ),
        )
        events.sort(key=lambda ev: ev[0])
        return events


class Sim:
    def __init__(self, spec: Spec, rng: random.Random, t_max: float = 180.0, plan: EventPlan | None = None):
        R = rules_for(spec)
        self.spec = spec
        self.R = R
        self.reseed(rng.getrandbits(64))
        self.t = 0.0
        self.t_max = t_max
        self.plan = plan or EventPlan()
        self.events: list[tuple[float, str, tuple]] = []
        self.ev_i = 0
        self.enemies: list[Enemy] = []
        self.next_enemy_id = 0
        self.has_boss = False
        self.tank_x, self.tank_y = 0.0, 0.0
        # player
        self.px, self.py = 0.0, 4.5
        self.knocked_until = -1.0
        self.res = [r.start for r in spec.resources]
        self.res_target = [-1] * R.n_res
        self.next_tick = [rng.uniform(0.0, r.tick_interval) if r.tick_interval > 0 else INF for r in spec.resources]
        self.target = -1
        self.gcd_until = 0.0
        self.casting = -1
        self.cast_target = -1
        self.cast_until = -1.0
        self.queued = -1
        self.swing = [0.0, rng.uniform(0.1, 0.5) if spec.off_hand else INF]
        self.cd_until = [0.0] * R.n_abilities
        self.group_until = [0.0] * R.n_groups
        self.buff_until = [-1.0] * len(R.buffs)
        self.buff_stacks = [0] * len(R.buffs)
        self.buff_charges = [0] * len(R.buffs)
        self.damage = 0.0
        # Expected damage: each strike adds its mean over the hit and crit rolls given the state before
        # the roll. Same expectation as `damage` with far less noise, so the teacher scores with it.
        self.ev_damage = 0.0
        self.dmg_by = [0.0] * (R.n_abilities + len(R.procs) + 1)  # abilities, procs, white
        self.done = False
        self.moving = True
        self.last_event: dict[str, float] = {}  # when each kind of scripted event last happened
        self.log: list[tuple[float, str]] | None = None
        # (time, ability index, target id) of each successful cast, when enabled: what the game tells
        # an addon in combat (UNIT_SPELLCAST_SUCCEEDED for the player)
        self.cast_log: list[tuple[float, int, int]] | None = None

    # ------------------------------------------------------------------ setup

    def add_enemy(self, name: str, x: float, y: float, hp: float, party_dps: float, boss: bool = False) -> Enemy:
        e = Enemy(self.next_enemy_id, name, boss, x, y, hp, party_dps)
        self.next_enemy_id += 1
        self.enemies.append(e)
        self.has_boss = self.has_boss or boss
        self._face_tank(e)
        if self.target < 0:
            self.target = e.id
        return e

    def schedule(self, events: list[tuple[float, str, tuple]]) -> None:
        self.events = events
        self.ev_i = 0

    def clone(self, seed: int | None = None) -> Sim:
        """An independent copy. Without a seed it replays the rolls the original would get; with one it
        gets fresh combat randomness, as from `reseed`."""
        s = Sim.__new__(Sim)
        s.__dict__.update(self.__dict__)
        s.enemies = [e.clone() for e in self.enemies]
        s.res = list(self.res)
        s.res_target = list(self.res_target)
        s.next_tick = list(self.next_tick)
        s.swing = list(self.swing)
        s.cd_until = list(self.cd_until)
        s.group_until = list(self.group_until)
        s.buff_until = list(self.buff_until)
        s.buff_stacks = list(self.buff_stacks)
        s.buff_charges = list(self.buff_charges)
        s.dmg_by = list(self.dmg_by)
        s.last_event = dict(self.last_event)
        s.log = None
        s.cast_log = None
        if seed is None:
            s.streams = []
            for r in self.streams:
                c = random.Random(0)
                c.setstate(r.getstate())
                s.streams.append(c)
        else:
            s.reseed(seed)
        return s

    def reseed(self, seed: int) -> None:
        """Replace all future combat rolls with fresh ones derived from seed."""
        self.streams = [random.Random(seed * N_STREAMS + i) for i in range(N_STREAMS)]

    def resample_future(self, rng: random.Random) -> None:
        """Replace upcoming scripted events with a fresh draw from the scenario's plan."""
        self.events = self.plan.sample(rng, self.t, self.t_max, self.last_event)
        self.ev_i = 0

    # ------------------------------------------------------------------ queries

    def enemy(self, eid: int) -> Enemy | None:
        for e in self.enemies:
            if e.id == eid:
                return e
        return None

    def dist(self, e: Enemy) -> float:
        return math.hypot(e.x - self.px, e.y - self.py)

    def facing(self, e: Enemy) -> tuple[float, float]:
        if e.turn_until > self.t:
            dx, dy = self.px - e.x, self.py - e.y
            n = math.hypot(dx, dy)
            if n > 1e-6:
                return dx / n, dy / n
        return e.fx, e.fy

    def behind(self, e: Enemy) -> bool:
        fx, fy = self.facing(e)
        return fx * (self.px - e.x) + fy * (self.py - e.y) < 0.0

    def buff_up(self, bi: int) -> bool:
        return self.buff_until[bi] > self.t

    def debuff(self, e: Enemy, di: int) -> tuple[float, int, int] | None:
        d = e.debuffs.get(di)
        if d is None or d[0] <= self.t:
            return None
        return d

    def points(self, ri: int, e: Enemy | None) -> float:
        """Amount of a resource usable against e (target-bound resources only count on their target)."""
        if ri in self.R.on_target:
            return self.res[ri] if e is not None and self.res_target[ri] == e.id else 0.0
        return self.res[ri]

    def cost_of(self, ai: int) -> list[tuple[int, float]]:
        out = []
        for r, amt in self.R.cost[ai]:
            mult = 1.0
            for bi, mults in self.R.cost_mult_buffs:
                if self.buff_until[bi] > self.t:
                    for k, m in mults:
                        if k == ALL or k == r:
                            mult *= m
            out.append((r, amt * mult))
        return out

    def affordable(self, ai: int) -> bool:
        for r, amt in self.cost_of(ai):
            if self.res[r] + EPS < amt:
                return False
        return True

    def haste(self) -> float:
        h = 1.0
        for bi, v in self.R.haste_buffs:
            if self.buff_until[bi] > self.t:
                h *= 1.0 + v
        return h

    def damage_mult(self, e: Enemy, school: str) -> float:
        m = 1.0
        for bi, v in self.R.dmg_buffs:
            if self.buff_until[bi] > self.t:
                m += v
        bonus = 0.0
        if e.debuffs:
            for di, (until, stacks, _) in e.debuffs.items():
                if until > self.t:
                    bonus += self.R.debuffs[di].damage_taken.get(school, 0.0) * stacks
        return m * (1.0 + bonus)

    def crit_chance(self, extra: float = 0.0) -> float:
        c = self.spec.crit_chance + extra
        for bi, v in self.R.crit_buffs:
            if self.buff_until[bi] > self.t:
                c += v
        return min(1.0, c)

    def in_range(self, ai: int, e: Enemy, d: float | None = None) -> bool:
        a = self.R.abilities[ai]
        d = self.dist(e) if d is None else d
        if a.gap_closer:
            return a.range_min <= d <= a.range_max
        return d <= (a.range_max or self.spec.melee_range)

    def legal_options(self) -> list[tuple[int, int]]:
        if self.done:
            return []
        opts: list[tuple[int, int]] = [(POOL, -1)]
        t = self.t
        if self.knocked_until > t:
            return opts
        R = self.R
        geo = None
        for ai, a in enumerate(R.abilities):
            if self.cd_until[ai] > t + EPS:
                continue
            g = R.group[ai]
            if g >= 0 and self.group_until[g] > t + EPS:
                continue
            if R.req_buff[ai] >= 0 and not self.buff_until[R.req_buff[ai]] > t:
                continue
            if R.req_no_buff[ai] >= 0 and self.buff_until[R.req_no_buff[ai]] > t:
                continue
            if not self.affordable(ai):
                continue
            if a.next_swing and self.queued >= 0:
                continue
            fin = R.finisher[ai]
            if not R.targeted[ai]:
                if fin >= 0 and self.res[fin] < 1:
                    continue
                if a.aoe and not any(self.dist(e) <= a.damage.aoe_radius for e in self.enemies):
                    continue
                opts.append((ai, -1))
                continue
            if geo is None:
                geo = [(e, self.dist(e)) for e in self.enemies]
            for e, d in geo:
                if not self.in_range(ai, e, d):
                    continue
                if a.req_behind and not self.behind(e):
                    continue
                if a.req_target_health_below and e.hp >= a.req_target_health_below * e.hp_max:
                    continue
                if a.req_target_health_above and e.hp <= a.req_target_health_above * e.hp_max:
                    continue
                if fin >= 0 and self.points(fin, e) < 1:
                    continue
                opts.append((ai, e.id))
        return opts

    # ------------------------------------------------------------------ actions

    def step(self, opt: tuple[int, int]) -> None:
        """Take an option and let time run to the next decision point: the end of the GCD or cast, or
        for POOL the next swing or resource tick (at most 1 s)."""
        if opt[0] == POOL:
            wait_to = min(self.swing[MH], self.swing[OH], min(self.next_tick), self.t + 1.0)
            self._run_until(max(wait_to, self.t + 0.05))
            return
        self.act(opt)
        if not self.done:
            end = max(self.gcd_until, self.cast_until if self.casting >= 0 else 0.0)
            if end > self.t:
                self._run_until(end)

    def act(self, opt: tuple[int, int]) -> None:
        """Start an option without letting time pass: queue a next-swing ability, begin a cast, or use an
        ability and start its GCD. A real-time front end calls this and runs the clock with `advance`."""
        ai, tid = opt
        if ai == POOL:
            return
        a = self.R.abilities[ai]
        if tid >= 0:
            self.target = tid
        if a.next_swing:
            self.queued = ai
            return
        if a.cast_time > 0:
            self.casting, self.cast_target, self.cast_until = ai, tid, self.t + a.cast_time
            self.gcd_until = self.t + self.R.gcd[ai]
            return
        self._use(ai, self.enemy(tid) if tid >= 0 else None)
        self._sweep()
        if self._check_done():
            return
        if self.R.gcd[ai] > 0:
            self.gcd_until = self.t + self.R.gcd[ai]

    def advance(self, t: float) -> None:
        """Let time run until t (or the end of the fight) without new actions."""
        self._run_until(t)

    def _use(self, ai: int, e: Enemy | None) -> None:
        """Pay for and resolve an ability."""
        R = self.R
        a = R.abilities[ai]
        t = self.t
        if self.log is not None:
            self.log.append((t, a.name + (f" -> {e.name}" if e is not None else "")))
        if self.cast_log is not None:
            self.cast_log.append((t, ai, e.id if e is not None else -1))
        for r, amt in self.cost_of(ai):
            self.res[r] = max(0.0, self.res[r] - amt)
        if R.has_cost[ai]:
            for bi in R.cost_consumed:
                if self.buff_until[bi] > t:
                    self.buff_until[bi] = t
        for bi in R.ability_consumed[ai]:
            self.buff_until[bi] = t
        if R.consumes[ai] >= 0:
            self.buff_until[R.consumes[ai]] = t
        if a.cooldown:
            self.cd_until[ai] = t + a.cooldown
        if R.group[ai] >= 0:
            self.group_until[R.group[ai]] = t + a.cooldown
        u = self._rolls(ABILITY_STREAM)
        points = 0.0
        fin = R.finisher[ai]
        if fin >= 0:
            points = self.points(fin, e)
            self.res[fin] = 0.0
            self.res_target[fin] = -1
        landed = True
        if a.damage is not None:
            landed = self._strike(ai, a.damage, e, points, ai, 0, u)
        if landed:
            hit = [e] if e is not None else []
            if a.aoe:
                hit = self._aoe_targets(a.damage)
            if a.dot is not None:
                for x in hit:
                    self._apply_dot(x, ai, a.dot.ticks, a.dot.interval, a.dot.per_tick + a.dot.per_tick_per_point * points, a.dot.school)
            if R.debuff_of[ai] >= 0:
                for x in hit:
                    self._apply_debuff(x, R.debuff_of[ai])
            for r, amt in R.gain[ai]:
                self._gain(r, amt, e)
        for r, v in R.set[ai]:
            self.res[r] = min(R.res[r].max, v)
        if R.buff_of[ai] >= 0:
            self._apply_buff(R.buff_of[ai], points if fin >= 0 else None)
        if a.gap_closer and e is not None:
            self.px, self.py = e.x - e.fx * BEHIND_GAP, e.y - e.fy * BEHIND_GAP
        for pi in R.by_trigger.get("ability:" + a.id, ()):
            self._proc(pi, e, 0, None, u)

    def _rolls(self, stream: int) -> list[float]:
        r = self.streams[stream].random
        return [r() for _ in range(self.R.n_rolls)]

    def _strike(self, src: int, dmg: Damage, e: Enemy | None, points: float, ai: int, depth: int = 0,
                u: list[float] | None = None) -> bool:
        """Resolve a yellow (ability) attack. Returns whether the primary target was hit. `u` is the
        primary target's block of rolls; without one it draws from the nested stream."""
        R = self.R
        sp = self.spec
        if dmg.aoe_radius > 0:
            targets = self._aoe_targets(dmg)
        else:
            if e is None:
                return False
            targets = [e] + self._cleave_targets(e, dmg.extra_targets)
        base = dmg.weapon * sp.main_hand.damage + dmg.flat
        if dmg.by_points:
            base += dmg.by_points[min(int(points), len(dmg.by_points) - 1)]
        if ai >= 0 and R.extra[ai] is not None:
            r, mx, per = R.extra[ai]
            spare = min(self.res[r], mx)
            self.res[r] -= spare
            base += spare * per
        builder = ai >= 0 and R.builds_on_target[ai]
        miss, dodge, cm = sp.miss_chance, sp.dodge_chance, sp.crit_multiplier
        p_hit = max(0.0, 1.0 - miss - dodge)
        primary_hit = False
        for i, x in enumerate(targets):
            if i > 0:
                b = self._rolls(SPLASH_STREAM)
            elif u is None:
                b = self._rolls(NESTED_STREAM)
            else:
                b = u
            crit_p = self.crit_chance(dmg.crit_bonus) if dmg.can_crit else 0.0
            amount = base * self.damage_mult(x, dmg.school)
            hp = x.hp if x.hp > 0.0 else 0.0
            self.ev_damage += p_hit * ((1.0 - crit_p) * min(amount, hp) + crit_p * min(amount * cm, hp))
            roll = b[0]
            if roll < miss:
                continue
            if roll < miss + dodge:
                self._fire("dodge", x, depth, None, b)
                continue
            crit = b[1] < crit_p
            self._deal(x, amount * cm if crit else amount, src, dmg.school)
            if i == 0:
                primary_hit = True
            self._fire("yellow_hit", x, depth, None, b)
            self._fire("hit", x, depth, None, b)
            if crit:
                self._fire("crit", x, depth, None, b)
                self._fire("yellow_crit", x, depth, None, b)
                if builder and i == 0:
                    self._fire("builder_crit", x, depth, None, b)
        return primary_hit or dmg.aoe_radius > 0

    def _white(self, hand: int, depth: int = 0) -> None:
        sp = self.spec
        R = self.R
        tgt = self.enemy(self.target)
        if tgt is None or self.knocked_until > self.t or self.dist(tgt) > sp.melee_range:
            self.swing[hand] = self.t + 0.2
            return
        weapon = sp.main_hand if hand == MH else sp.off_hand
        if depth == 0:
            for bi in R.swing_consumed:
                if self.buff_until[bi] > self.t:
                    self.buff_charges[bi] -= 1
                    if self.buff_charges[bi] <= 0:
                        self.buff_until[bi] = self.t
            self.swing[hand] = self.t + weapon.speed / self.haste()
        u = self._rolls(hand if depth == 0 else NESTED_STREAM)
        if hand == MH and self.queued >= 0 and depth == 0:
            ai, self.queued = self.queued, -1
            if self.affordable(ai):
                a = R.abilities[ai]
                for r, amt in self.cost_of(ai):
                    self.res[r] = max(0.0, self.res[r] - amt)
                if self.log is not None:
                    self.log.append((self.t, a.name + f" -> {tgt.name}"))
                if self.cast_log is not None:
                    self.cast_log.append((self.t, ai, tgt.id))
                self._strike(ai, a.damage, tgt, 0.0, ai, 0, u)
                return
        miss = sp.miss_chance + (sp.dual_wield_miss if sp.off_hand else 0.0)
        dodge, glance_p, cm, gm = sp.dodge_chance, sp.glancing_chance, sp.crit_multiplier, sp.glancing_multiplier
        crit_p = self.crit_chance()
        amount = weapon.damage * self.damage_mult(tgt, "physical")
        hp = tgt.hp if tgt.hp > 0.0 else 0.0
        regular = max(0.0, 1.0 - miss - dodge - glance_p)
        self.ev_damage += glance_p * min(amount * gm, hp) + regular * (
            (1.0 - crit_p) * min(amount, hp) + crit_p * min(amount * cm, hp))
        roll = u[0]
        if roll < miss:
            return
        roll -= miss
        if roll < dodge:
            self._fire("dodge", tgt, depth, None, u)
            return
        roll -= dodge
        glance = roll < glance_p
        crit = not glance and u[1] < crit_p
        if glance:
            amount *= gm
        elif crit:
            amount *= cm
        self._deal(tgt, amount, len(self.dmg_by) - 1, "physical")
        for r, per in R.white_gain:
            self.res[r] = min(R.res[r].max, self.res[r] + amount * per)
        self._fire("white_hit", tgt, depth, weapon.speed, u)
        self._fire("hit", tgt, depth, weapon.speed, u)
        if crit:
            self._fire("crit", tgt, depth, weapon.speed, u)
            self._fire("white_crit", tgt, depth, weapon.speed, u)

    def _fire(self, trigger: str, e: Enemy, depth: int, speed: float | None = None,
              u: list[float] | None = None) -> None:
        for pi in self.R.by_trigger.get(trigger, ()):
            self._proc(pi, e, depth, speed, u)

    def _proc(self, pi: int, e: Enemy | None, depth: int, speed: float | None = None,
              u: list[float] | None = None) -> None:
        R = self.R
        p = R.procs[pi]
        if R.proc_req[pi] >= 0 and not self.buff_until[R.proc_req[pi]] > self.t:
            return
        chance = p.chance if not p.ppm else p.ppm * (speed or self.spec.main_hand.speed) / 60.0
        if chance < 1.0:
            roll = u[2 + pi] if u is not None else self.streams[NESTED_STREAM].random()
            if roll >= chance:
                return
        if R.proc_buff[pi] >= 0:
            self._apply_buff(R.proc_buff[pi], None)
        for r, amt in R.proc_gain[pi]:
            self._gain(r, amt, e)
        if e is None or depth >= MAX_PROC_DEPTH or e.hp <= 0:
            return
        if R.proc_debuff[pi] >= 0:
            self._apply_debuff(e, R.proc_debuff[pi])
        if p.dot is not None:
            self._apply_dot(e, R.n_abilities + pi, p.dot.ticks, p.dot.interval, p.dot.per_tick, p.dot.school)
        if p.damage is not None:
            self._strike(R.n_abilities + pi, p.damage, e, 0.0, -1, depth + 1)
        for _ in range(p.extra_attacks):
            self._white(MH, depth + 1)

    def _gain(self, r: int, amt: float, e: Enemy | None) -> None:
        if r in self.R.on_target:
            if e is None:
                return
            if self.res_target[r] != e.id:
                self.res[r], self.res_target[r] = 0.0, e.id
        self.res[r] = min(self.R.res[r].max, self.res[r] + amt)

    def _apply_buff(self, bi: int, points: float | None) -> None:
        b: Aura = self.R.buffs[bi]
        dur = b.duration
        if b.duration_by_points and points is not None:
            dur = b.duration_by_points[min(int(points), len(b.duration_by_points) - 1)]
        active = self.buff_until[bi] > self.t
        self.buff_stacks[bi] = min(b.max_stacks, self.buff_stacks[bi] + 1) if active else 1
        self.buff_charges[bi] = b.charges
        self.buff_until[bi] = self.t + dur

    def _apply_debuff(self, e: Enemy, di: int) -> None:
        d: Aura = self.R.debuffs[di]
        cur = self.debuff(e, di)
        stacks = min(d.max_stacks, cur[1] + 1) if cur else 1
        e.debuffs[di] = (self.t + d.duration, stacks, d.charges)

    def _apply_dot(self, e: Enemy, key: int, ticks: int, interval: float, per_tick: float, school: str) -> None:
        e.dots[key] = (self.t + interval, ticks, per_tick, interval, school)

    def _aoe_targets(self, dmg: Damage) -> list[Enemy]:
        near = sorted((self.dist(e), e.id, e) for e in self.enemies)
        return [e for d, _, e in near if d <= dmg.aoe_radius][: dmg.max_targets]

    def _cleave_targets(self, e: Enemy, extra: int) -> list[Enemy]:
        radius = 8.0
        for bi, n, r in self.R.cleave_buffs:
            if self.buff_until[bi] > self.t:
                extra += n
                radius = max(radius, r)
        if extra <= 0:
            return []
        others = sorted(
            (math.hypot(o.x - e.x, o.y - e.y), o.id, o) for o in self.enemies if o is not e
        )
        return [o for d, _, o in others if d <= radius][:extra]

    def _deal(self, e: Enemy, dmg: float, src: int, school: str) -> None:
        eff = dmg if dmg < e.hp else max(0.0, e.hp)
        e.hp -= dmg
        e.pending += eff
        self.damage += eff
        self.dmg_by[src] += eff
        consumed = self.R.school_consumed.get(school)
        if consumed and e.debuffs:
            for di in consumed:
                cur = e.debuffs.get(di)
                if cur and cur[0] > self.t and cur[2] > 0:
                    e.debuffs[di] = (cur[0] if cur[2] > 1 else self.t, cur[1], cur[2] - 1)

    # ------------------------------------------------------------------ time

    def _run_until(self, t_end: float) -> None:
        t_end = min(t_end, self.t_max)
        while not self.done:
            t = self.t
            nxt = t_end
            for nt in self.next_tick:
                if nt < nxt:
                    nxt = nt
            if self.swing[MH] < nxt:
                nxt = self.swing[MH]
            if self.swing[OH] < nxt:
                nxt = self.swing[OH]
            if self.casting >= 0 and self.cast_until < nxt:
                nxt = self.cast_until
            for e in self.enemies:
                for d in e.dots.values():
                    if d[0] < nxt:
                        nxt = d[0]
                if e.engaged and e.party_dps > 0:
                    death = t + e.hp / e.party_dps
                    if death < nxt:
                        nxt = death
            if self.ev_i < len(self.events) and self.events[self.ev_i][0] < nxt:
                nxt = self.events[self.ev_i][0]
            if self.moving and nxt - t > MOVE_STEP:
                nxt = t + MOVE_STEP
            if nxt < t:
                nxt = t
            self._integrate(nxt - t)
            self.t = nxt
            self._discrete()
            self._sweep()
            if self._check_done() or self.t >= t_end - EPS:
                return

    def _integrate(self, dt: float) -> None:
        if dt <= 0:
            return
        R = self.R
        for r in R.flowing:
            self.res[r] = min(R.res[r].max, self.res[r] + R.res[r].regen_per_sec * self._regen_mult(r) * dt)
        decay = math.exp(-dt / LOSS_TAU)
        moving = False
        tx, ty = self.tank_x, self.tank_y
        for e in self.enemies:
            dx, dy = e.x - tx, e.y - ty
            dtank = math.hypot(dx, dy)
            if dtank > TANK_GAP + 0.05:
                step = min(e.speed * dt, dtank - TANK_GAP)
                e.x -= dx / dtank * step
                e.y -= dy / dtank * step
                moving = True
                self._face_tank(e)
            elif not e.engaged:
                e.engaged = True
            party = e.party_dps if e.engaged else 0.0
            if party:
                e.hp -= party * dt
            e.loss_rate = e.loss_rate * decay + party * (1.0 - decay) + e.pending / LOSS_TAU
            e.pending = 0.0
        if self.knocked_until <= self.t:
            tgt = self.enemy(self.target)
            if tgt is not None:
                gx, gy = tgt.x - tgt.fx * BEHIND_GAP, tgt.y - tgt.fy * BEHIND_GAP
                dx, dy = gx - self.px, gy - self.py
                d = math.hypot(dx, dy)
                if d > 0.05:
                    step = min(self.spec.run_speed * dt, d)
                    self.px += dx / d * step
                    self.py += dy / d * step
                    moving = moving or d - step > 0.05
        else:
            moving = True
        self.moving = moving

    def _regen_mult(self, r: int) -> float:
        m = 1.0
        for bi, mults in self.R.regen_mult_buffs:
            if self.buff_until[bi] > self.t:
                for k, v in mults:
                    if k == r:
                        m *= v
        return m

    def _discrete(self) -> None:
        R = self.R
        t = self.t + EPS
        for r in R.ticked:
            while self.next_tick[r] <= t:
                res = R.res[r]
                self.res[r] = min(res.max, self.res[r] + res.tick_amount * self._regen_mult(r))
                self.next_tick[r] += res.tick_interval
        for e in self.enemies:
            if not e.dots:
                continue
            for key, (nt, left, dmg, iv, school) in list(e.dots.items()):
                if nt <= t:
                    amount = dmg * self.damage_mult(e, school)
                    self.ev_damage += min(amount, e.hp) if e.hp > 0.0 else 0.0
                    self._deal(e, amount, key, school)
                    if left > 1:
                        e.dots[key] = (nt + iv, left - 1, dmg, iv, school)
                    else:
                        del e.dots[key]
        if self.casting >= 0 and self.cast_until <= t:
            ai, tid = self.casting, self.cast_target
            self.casting = -1
            e = self.enemy(tid) if tid >= 0 else None
            if (tid < 0 or (e is not None and self.in_range(ai, e))) and self.affordable(ai):
                self._use(ai, e)
        for hand in (MH, OH):
            if self.swing[hand] <= t:
                self._white(hand)
        while self.ev_i < len(self.events) and self.events[self.ev_i][0] <= t:
            _, kind, payload = self.events[self.ev_i]
            self.ev_i += 1
            self._event(kind, payload)

    def _event(self, kind: str, payload: tuple) -> None:
        self.last_event[kind] = self.t
        boss = next((e for e in self.enemies if e.boss), None) or (self.enemies[0] if self.enemies else None)
        if kind == "knockback" and boss is not None:
            dx, dy = self.px - boss.x, self.py - boss.y
            n = math.hypot(dx, dy) or 1.0
            self.px = boss.x + dx / n * payload[0]
            self.py = boss.y + dy / n * payload[0]
            self.knocked_until = self.t + KNOCKED_TIME
            self.casting = -1
            self.moving = True
            if self.log is not None:
                self.log.append((self.t, f"[knockback {payload[0]:.0f} yd]"))
        elif kind == "turn" and boss is not None:
            boss.turn_until = self.t + payload[0]
            if self.log is not None:
                self.log.append((self.t, f"[{boss.name} turns for {payload[0]:.1f}s]"))
        elif kind == "reposition":
            ang, dist = payload
            self.tank_x += math.cos(ang) * dist
            self.tank_y += math.sin(ang) * dist
            self.moving = True
            for e in self.enemies:
                self._face_tank(e)
            if self.log is not None:
                self.log.append((self.t, f"[tank repositions {dist:.0f} yd]"))
        elif kind == "spawn":
            for hp, party, ang in payload:
                x = self.tank_x + math.cos(ang) * 22.0
                y = self.tank_y + math.sin(ang) * 22.0
                self.add_enemy(f"Add {self.next_enemy_id}", x, y, hp, party)
            self.moving = True
            if self.log is not None:
                self.log.append((self.t, f"[{len(payload)} adds spawn]"))

    def _face_tank(self, e: Enemy) -> None:
        dx, dy = self.tank_x - e.x, self.tank_y - e.y
        n = math.hypot(dx, dy)
        if n > 0.3:
            e.fx, e.fy = dx / n, dy / n

    def _sweep(self) -> None:
        if not any(e.hp <= 1e-6 for e in self.enemies):
            return
        for e in [e for e in self.enemies if e.hp <= 1e-6]:
            self.enemies.remove(e)
            if self.log is not None:
                self.log.append((self.t, f"[{e.name} dies]"))
            for r in self.R.on_target:
                if self.res_target[r] == e.id:
                    self.res[r], self.res_target[r] = 0.0, -1
            if self.target == e.id:
                near = min(self.enemies, key=self.dist, default=None)
                self.target = near.id if near is not None else -1

    def _check_done(self) -> bool:
        if self.t >= self.t_max - EPS:
            self.done = True
        elif self.has_boss:
            self.done = not any(e.boss for e in self.enemies)
        else:
            pending_spawns = any(k == "spawn" for _, k, _ in self.events[self.ev_i :])
            self.done = not self.enemies and not pending_spawns
        return self.done
