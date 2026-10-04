"""What the in-game addon can know in combat, rebuilt from the player's own casts.

On Forever, an addon gets your successful casts, target changes and deaths, range checks and the
clock in combat. Energy, combo points, buffs, cooldown values and target health are drawn by the game
but hidden from code. The tracker rebuilds what it can from those events and the spec's rules:
- cooldowns and the GCD, exactly
- your own DoTs and debuffs on the current target, and your own buffs, from when you cast them
- each pooled resource (energy, mana) as an estimate: its value at the pull, average regeneration,
  and the costs and gains of your casts; procs it can't see (a free cast from Clearcasting) make the
  estimate drift low, so it's clamped at zero
- target-bound points (combo points) as the number of builders since your last finisher on this
  target; crits that add extra points are invisible

The Lua addon mirrors this module operation for operation, and `features` is the policy's only input.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim import POOL, rules_for
from ..spec import Spec

T = 10.0  # time unit for features
RECENT = 30.0  # cap on "seconds since"
GATED_KINDS = ("energy", "rage")  # the main spending resource: the game can gate frames on its value
BANDS = 5  # the addon precomputes one recommendation per band; the game shows the one matching your
           # real (hidden) value via UnitPowerPercent with a step curve, so the addon never reads it


class Tracker:
    def __init__(self, spec: Spec):
        R = rules_for(spec)
        self.spec, self.R = spec, R
        self.A = R.n_abilities
        # per-ability static data the addon ships with
        self.dot_abilities = [ai for ai, a in enumerate(R.abilities) if a.dot is not None]
        self.debuff_abilities = [ai for ai in range(self.A) if R.debuff_of[ai] >= 0]
        self.buffs = sorted({R.buff_of[ai] for ai in range(self.A) if R.buff_of[ai] >= 0})
        self.pooled = [r for r in range(R.n_res) if r not in R.on_target]
        self.points = [r for r in range(R.n_res) if r in R.on_target]
        self.regen = []
        for r, res in enumerate(R.res):
            rate = res.regen_per_sec + (res.tick_amount / res.tick_interval if res.tick_interval > 0 else 0.0)
            self.regen.append(rate)
        self.gated = next((r for r in self.pooled if R.res[r].kind in GATED_KINDS), -1)
        self.width = len(self.features_layout())

    def bands(self) -> list[float]:
        """Lower edges of the gated resource's bands, as amounts."""
        if self.gated < 0:
            return []
        mx = self.R.res[self.gated].max
        return [mx * i / BANDS for i in range(BANDS)]

    def band_of(self, value: float) -> float:
        return max(b for b in self.bands() if b <= value + 1e-9)

    # ------------------------------------------------------------------ events

    def start(self, t: float, values: list[float]) -> None:
        """The pull: resource values the addon read just before combat."""
        R = self.R
        self.t0 = self.now = t
        self.est = list(values)
        self.ready = [t] * self.A
        self.group_ready = [t] * R.n_groups
        self.gcd_ready = t
        self.last = [-math.inf] * self.A
        self.last_ability = -1
        self.buff_until = {bi: t for bi in self.buffs}
        self.retarget(t)

    def retarget(self, t: float) -> None:
        """Target changed or died: per-target knowledge starts over."""
        self.target_since = t
        self.dot_until = {ai: t for ai in self.dot_abilities}
        self.debuff_until = {ai: t for ai in self.debuff_abilities}
        self.built = [0] * len(self.points)  # builders since the last finisher, per target-bound resource

    def advance(self, t: float) -> None:
        dt = t - self.now
        if dt > 0:
            for r in self.pooled:
                self.est[r] = min(self.R.res[r].max, self.est[r] + self.regen[r] * self._regen_mult(r) * dt)
            self.now = t

    def cast(self, t: float, ai: int) -> None:
        """You cast ability ai (the game's UNIT_SPELLCAST_SUCCEEDED)."""
        R = self.R
        self.advance(t)
        for r, amt in R.cost[ai]:
            if r in self.pooled:
                self.est[r] = max(0.0, self.est[r] - amt * self._cost_mult(r))
        fin = R.finisher[ai]
        spent = 0
        if fin >= 0 and fin in self.points:
            i = self.points.index(fin)
            spent = self.built[i]
            self.built[i] = 0
        for r, amt in R.gain[ai]:
            if r in self.pooled:
                self.est[r] = min(R.res[r].max, self.est[r] + amt)
            else:
                i = self.points.index(r)
                self.built[i] += 1
        for r, v in R.set[ai]:
            if r in self.pooled:
                self.est[r] = min(R.res[r].max, v)
        a = R.abilities[ai]
        if a.cooldown:
            self.ready[ai] = t + a.cooldown
            if R.group[ai] >= 0:
                self.group_ready[R.group[ai]] = t + a.cooldown
        if R.gcd[ai] > 0:
            self.gcd_ready = t + R.gcd[ai]
        bi = R.buff_of[ai]
        if bi >= 0:
            b = R.buffs[bi]
            dur = b.duration_by_points[min(spent, len(b.duration_by_points) - 1)] if b.duration_by_points else b.duration
            self.buff_until[bi] = t + dur
        if a.dot is not None:
            self.dot_until[ai] = t + a.dot.ticks * a.dot.interval
        if R.debuff_of[ai] >= 0:
            self.debuff_until[ai] = t + R.debuffs[R.debuff_of[ai]].duration
        self.last[ai] = t
        self.last_ability = ai

    def _cost_mult(self, r: int, t: float | None = None) -> float:
        t = self.now if t is None else t
        m = 1.0
        for bi, mults in self.R.cost_mult_buffs:
            if bi in self.buff_until and self.buff_until[bi] > t:
                for k, v in mults:
                    if k == -1 or k == r:
                        m *= v
        return m

    def _regen_mult(self, r: int) -> float:
        m = 1.0
        for bi, mults in self.R.regen_mult_buffs:
            if bi in self.buff_until and self.buff_until[bi] > self.now:
                for k, v in mults:
                    if k == r:
                        m *= v
        return m

    def estimates(self, t: float, known: float | None = None) -> list[float]:
        """Resource estimates projected to time t, without changing the tracker. `known` is the gated
        resource's value at time t under the hypothesis being evaluated (a band's lower edge)."""
        dt = max(0.0, t - self.now)
        est = list(self.est)
        for r in self.pooled:
            est[r] = min(self.R.res[r].max, est[r] + self.regen[r] * self._regen_mult(r) * dt)
        if known is not None and self.gated >= 0:
            est[self.gated] = known
        return est

    def copy(self) -> Tracker:
        c = Tracker.__new__(Tracker)
        c.__dict__.update(self.__dict__)
        for k in ("est", "ready", "group_ready", "last", "built"):
            setattr(c, k, list(getattr(self, k)))
        for k in ("buff_until", "dot_until", "debuff_until"):
            setattr(c, k, dict(getattr(self, k)))
        return c

    def predict(self, scores, n: int, in_melee: bool, in_charge_range: bool, known: float | None = None,
                wait_step: float = 0.5, max_wait: float = 3.0) -> list[tuple[int, float]]:
        """The next n spells if you follow the recommendations: (action, seconds from now until it's due).
        `scores(features, possible)` returns the policy's scores per action. `known` is the gated
        resource's value now (a band hypothesis). A "wait" advances a copy of the tracker by wait_step
        (at most max_wait in a row) instead of adding to the queue."""
        c = self.copy()
        start = t = max(c.now, c.gcd_ready)
        if known is not None and c.gated >= 0:
            c.advance(t)
            c.est[c.gated] = known
        out, waited = [], 0.0
        while len(out) < n:
            f, m = c.features(in_melee, in_charge_range), c.possible(in_melee, in_charge_range)
            s = scores(f, m)
            a = max((i for i in range(len(m)) if m[i]), key=lambda i: s[i])
            if a == self.A and waited < max_wait:
                t += wait_step
                waited += wait_step
                c.advance(t)
                c.gcd_ready = max(c.gcd_ready, t)
                continue
            if a == self.A:
                break
            out.append((a, t - start))
            c.cast(t, a)
            t = max(c.now, c.gcd_ready)
            waited = 0.0
        return out

    # ------------------------------------------------------------------ what the policy sees

    def features_layout(self) -> list[str]:
        R = self.R
        names = []
        for a in R.abilities:
            names += [f"{a.id}.cooldown", f"{a.id}.ready", f"{a.id}.since", f"{a.id}.affordable"]
        names += [f"{R.abilities[ai].id}.dot" for ai in self.dot_abilities]
        names += [f"{R.abilities[ai].id}.debuff" for ai in self.debuff_abilities]
        names += [f"{R.buffs[bi].id}.buff" for bi in self.buffs]
        names += [f"{R.res_ids[r]}.estimate" for r in self.pooled]
        names += [f"{R.res_ids[r]}.builders" for r in self.points]
        names += ["combat_time", "target_time", "in_melee", "in_charge_range"]
        names += [f"last.{a.id}" for a in R.abilities]
        return names

    def features(self, in_melee: bool, in_charge_range: bool, known: float | None = None) -> np.ndarray:
        """As of the moment the GCD ends (the next decision), from what the addon knows, with the gated
        resource set to `known` when given. Read-only."""
        R, t = self.R, max(self.now, self.gcd_ready)
        est = self.estimates(t, known)
        out = []
        for ai, a in enumerate(R.abilities):
            g = R.group[ai]
            cd = max(0.0, self.ready[ai] - t, (self.group_ready[g] - t) if g >= 0 else 0.0)
            out += [math.log1p(cd / T), float(cd <= 1e-9), min(RECENT, t - self.last[ai]) / T,
                    float(self.affordable(ai, est, t))]
        out += [max(0.0, self.dot_until[ai] - t) / T for ai in self.dot_abilities]
        out += [max(0.0, self.debuff_until[ai] - t) / T for ai in self.debuff_abilities]
        out += [max(0.0, self.buff_until[bi] - t) / T for bi in self.buffs]
        out += [est[r] / R.res[r].max for r in self.pooled]
        out += [min(self.built[i], 2 * R.res[r].max) / R.res[r].max for i, r in enumerate(self.points)]
        out += [min(t - self.t0, 300.0) / 100.0, min(RECENT, t - self.target_since) / T, float(in_melee),
                float(in_charge_range)]
        out += [float(self.last_ability == ai) for ai in range(self.A)]
        return np.asarray(out, dtype=np.float32)

    def affordable(self, ai: int, est: list[float], t: float) -> bool:
        return all(est[r] + 1e-9 >= amt * self._cost_mult(r, t) for r, amt in self.R.cost[ai] if r in self.pooled)

    def possible(self, in_melee: bool, in_charge_range: bool) -> np.ndarray:
        """Actions that are certainly impossible at the next decision are masked: cooldown not ready, a
        finisher with nothing built, or out of range. Energy isn't masked; the estimate can be wrong.
        The last entry is "wait"."""
        R, t = self.R, max(self.now, self.gcd_ready)
        ok = np.ones(self.A + 1, dtype=bool)
        for ai, a in enumerate(R.abilities):
            g = R.group[ai]
            if self.ready[ai] > t + 1e-9 or (g >= 0 and self.group_ready[g] > t + 1e-9):
                ok[ai] = False
            fin = R.finisher[ai]
            if fin >= 0 and fin in self.points and self.built[self.points.index(fin)] == 0:
                ok[ai] = False
            if a.gap_closer:
                ok[ai] &= in_charge_range
            elif R.targeted[ai] and not in_melee and not a.range_max:
                ok[ai] = False
        return ok


def action_of(opt: tuple[int, int], target: int, n_abilities: int) -> int:
    """The addon's action for a teacher option: its ability, or "wait". Options on other targets map to
    -1: the addon recommends spells for your current target, and you pick targets."""
    ai, tid = opt
    if ai == POOL:
        return n_abilities
    return ai if tid in (-1, target) else -1
