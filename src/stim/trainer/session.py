"""Practice-trainer game logic: a fight on a real (scalable) clock that you play with the spec's keys,
coached by the model, and graded afterwards. Kept free of networking so it can be tested directly."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .. import features as F
from ..model import Batch
from ..policies import best_baseline
from ..scenarios import make_scenario
from ..sim import EPS, POOL, Sim
from ..spec import load_spec
from ..teacher import Teacher, TeacherConfig

Option = tuple[int, int]
QUEUE_WINDOW = 0.4  # seconds before the GCD ends in which a press is queued, like the game's spell queue
IDLE_GRACE = 1.0  # waiting this long with something usable counts as a decision to pool
STAKES = ("low", "medium", "high", "critical")


@dataclass
class Decision:
    t: float
    state: Sim  # clone of the moment the choice was made (or the GCD came up, for idling)
    options: list[Option]
    chosen: Option
    advice: Option | None
    delay: float  # seconds the GCD had been ready before the press


@dataclass
class Session:
    spec_name: str = "feral"
    kind: str = "boss"
    seed: int | None = None
    speed: float = 1.0
    student: object | None = None  # a StudentPolicy, for live advice and the model's run of the same pull
    decisions: list[Decision] = field(default_factory=list)

    def __post_init__(self):
        self.seed = random.randrange(1 << 30) if self.seed is None else self.seed
        self.spec = load_spec(self.spec_name)
        self.sim = make_scenario(self.spec, self.kind, random.Random(self.seed))
        self.keys = {a.key.lower(): i for i, a in enumerate(self.spec.abilities) if a.key}
        self.queued: Option | None = None
        self.ready_since: float | None = None
        self._ready_state: tuple[Sim, list[Option]] | None = None
        self.paused = False
        self.message = ""
        self.message_until = 0.0
        self._advice_cache: tuple[float, int, dict | None] = (-1.0, -1, None)

    # ------------------------------------------------------------------ input

    def press(self, key: str) -> str:
        """Use the ability bound to key on the current target. Returns "" or why it didn't happen."""
        sim = self.sim
        if sim.done:
            return "the fight is over"
        if self.paused:
            return self._say("paused")
        ai = self.keys.get(key.lower())
        if ai is None:
            return ""
        opt = self._option(ai)
        gcd_left = sim.gcd_until - sim.t
        if self.spec.abilities[ai].off_gcd or self.spec.abilities[ai].next_swing or gcd_left <= EPS:
            return self._try(opt)
        if gcd_left <= QUEUE_WINDOW:
            self.queued = opt
            return ""
        return self._say("not ready yet")

    def target(self, eid: int) -> None:
        if self.sim.enemy(eid) is not None:
            self.sim.target = eid

    def cycle_target(self) -> None:
        sim = self.sim
        order = sorted(sim.enemies, key=sim.dist)
        if not order:
            return
        ids = [e.id for e in order]
        i = ids.index(sim.target) if sim.target in ids else -1
        sim.target = ids[(i + 1) % len(ids)]

    def _option(self, ai: int) -> Option:
        return (ai, self.sim.target) if self.sim.R.targeted[ai] else (ai, -1)

    def _try(self, opt: Option) -> str:
        sim = self.sim
        opts = sim.legal_options()
        if opt not in opts:
            return self._say(why_not(sim, opt))
        advice = self.advice()
        self.decisions.append(Decision(sim.t, sim.clone(), opts, opt, advice["option"] if advice else None,
                                       sim.t - self.ready_since if self.ready_since is not None else 0.0))
        sim.act(opt)
        self.ready_since = None
        return ""

    def _say(self, text: str) -> str:
        self.message, self.message_until = text, self.sim.t + 1.5
        return text

    # ------------------------------------------------------------------ clock

    def tick(self, dt: float) -> None:
        """Advance by dt seconds of real time."""
        sim = self.sim
        if self.paused or sim.done:
            return
        end = sim.t + dt * self.speed
        if self.queued is not None and sim.gcd_until <= end:
            sim.advance(max(sim.t, sim.gcd_until))
            opt, self.queued = self.queued, None
            self._ready()
            self._try(opt)
        if not sim.done:
            sim.advance(end)
        self._ready()

    def _ready(self) -> None:
        """Track when the GCD came up, and log long idles as decisions to pool."""
        sim = self.sim
        ready = sim.gcd_until <= sim.t + EPS and sim.knocked_until <= sim.t and sim.casting < 0 and not sim.done
        if not ready:
            self.ready_since = None
            return
        if self.ready_since is None:
            self.ready_since = sim.t
            self._ready_state = (sim.clone(), sim.legal_options())
        elif sim.t - self.ready_since >= IDLE_GRACE and self._ready_state is not None:
            # every further second of waiting with something usable is another decision to wait
            snap, opts = self._ready_state
            if len(opts) > 1:
                self.decisions.append(Decision(snap.t, snap, opts, (POOL, -1), None, sim.t - snap.t))
            self.ready_since = sim.t
            self._ready_state = (sim.clone(), sim.legal_options())

    # ------------------------------------------------------------------ coaching

    @torch.inference_mode()
    def advice(self) -> dict | None:
        """The model's view of the next decision: its top option, the runners-up, its confidence that the
        teacher agrees, and how much the decision matters. While the GCD is still running it previews the
        moment the GCD ends, on a clone with fresh rolls so the preview can't know future luck."""
        if self.student is None or self.sim.done:
            return None
        key = (round(self.sim.t, 2), self.sim.target)
        if self._advice_cache[:2] == key:
            return self._advice_cache[2]
        sim = self.sim
        if sim.gcd_until > sim.t + EPS or sim.casting >= 0:
            sim = sim.clone(seed=1)
            sim.events = []
            sim.advance(max(sim.gcd_until, sim.cast_until if sim.casting >= 0 else 0.0))
        opts = sim.legal_options()
        out = None
        if len(opts) > 1:
            res = self.student.model(Batch([F.tokens(sim, opts)], self.student.device))
            p = res["choice"].softmax(-1)[0].tolist()
            order = sorted(range(len(opts)), key=lambda i: -p[i])
            stakes = int((torch.sigmoid(res["score"][0]) > 0.5).sum())
            out = {
                "option": opts[order[0]],
                "top": [(opts[i], p[i]) for i in order[:3]],
                "confidence": float(torch.sigmoid(res["conf"][0])),
                "stakes": STAKES[stakes],
                "survive": float(torch.sigmoid(res["noul"][0])),
            }
        self._advice_cache = (key[0], key[1], out)
        return out

    def model_run(self) -> float | None:
        """DPS the model gets on this exact pull (same seed, so the same script and the same luck)."""
        if self.student is None:
            return None
        sim = make_scenario(self.spec, self.kind, random.Random(self.seed))
        while not sim.done:
            sim.step(self.student(sim, sim.legal_options()))
        return sim.damage / sim.t

    def review(self, samples: int = 16, rollout=None, worst: int = 6) -> dict:
        """Grade every recorded decision with the teacher and list the costliest ones."""
        rollout = rollout or best_baseline(self.spec)
        teacher = Teacher(rollout, TeacherConfig(samples=samples))
        rng = random.Random(self.seed)
        ref = self.spec.reference_dps
        graded = []
        for d in self.decisions:
            label = teacher.evaluate(d.state, rng, d.options)
            i = d.options.index(d.chosen)
            graded.append({
                "t": round(d.t, 1),
                "chosen": describe(d.state, d.chosen),
                "best": describe(d.state, label.options[label.best]),
                "regret": float(label.gap[i] / ref),
                "delay": round(d.delay, 2),
                "followed_advice": d.advice == d.chosen if d.advice is not None else None,
                "context": context(d.state),
            })
        total = sum(g["regret"] for g in graded)
        return {
            "decisions": len(graded),
            "regret_total": total,
            "worst": sorted(graded, key=lambda g: -g["regret"])[:worst],
            "advice_followed": sum(1 for g in graded if g["followed_advice"]) / max(1, sum(1 for g in graded if g["followed_advice"] is not None)),
        }

    # ------------------------------------------------------------------ view

    def state(self) -> dict:
        sim, R, t = self.sim, self.sim.R, self.sim.t
        tgt = sim.enemy(sim.target)
        gcd_ready = sim.gcd_until <= t + EPS and sim.casting < 0
        legal = {ai for ai, _ in sim.legal_options() if ai >= 0 and (gcd_ready or R.gcd[ai] == 0.0)}
        adv = self.advice()
        return {
            "t": t,
            "done": sim.done,
            "paused": self.paused,
            "speed": self.speed,
            "spec": self.spec.name,
            "kind": self.kind,
            "seed": self.seed,
            "dps": sim.damage / t if t > 0 else 0.0,
            "damage": sim.damage,
            "message": self.message if t < self.message_until else "",
            "player": {"x": sim.px, "y": sim.py, "gcd": max(0.0, sim.gcd_until - t), "gcd_total": self.spec.gcd,
                       "knocked": sim.knocked_until > t, "queued": describe(sim, self.queued) if self.queued else ""},
            "tank": {"x": sim.tank_x, "y": sim.tank_y},
            "melee_range": self.spec.melee_range,
            "resources": [{"id": r.id, "kind": r.kind, "value": sim.res[i], "max": r.max,
                           "on_target": i in R.on_target, "holder": sim.res_target[i]} for i, r in enumerate(R.res)],
            "buffs": [{"id": b.id, "name": b.name, "remaining": sim.buff_until[i] - t, "stacks": sim.buff_stacks[i]}
                      for i, b in enumerate(R.buffs) if sim.buff_until[i] > t],
            "enemies": [{
                "id": e.id, "name": e.name, "boss": e.boss, "x": e.x, "y": e.y,
                "fx": sim.facing(e)[0], "fy": sim.facing(e)[1], "hp": e.hp, "hp_max": e.hp_max,
                "target": e.id == sim.target, "behind": sim.behind(e), "dist": sim.dist(e),
                "dots": [{"name": dot_name(sim, k), "remaining": nt - t + (left - 1) * iv}
                         for k, (nt, left, _, iv, _) in e.dots.items()],
                "debuffs": [{"name": R.debuffs[di].name, "remaining": until - t, "stacks": st}
                            for di, (until, st, _) in e.debuffs.items() if until > t],
            } for e in sim.enemies],
            "target": tgt.id if tgt else -1,
            "abilities": [{
                "id": a.id, "name": a.name, "key": a.key,
                "cooldown": max(0.0, sim.cd_until[i] - t), "cooldown_total": a.cooldown,
                "usable": i in legal, "off_gcd": a.off_gcd or a.next_swing,
                "advised": adv is not None and adv["option"][0] == i,
            } for i, a in enumerate(R.abilities)],
            "advice": None if adv is None else {
                "text": describe(sim, adv["option"]),
                "top": [{"text": describe(sim, o), "p": p} for o, p in adv["top"]],
                "confidence": adv["confidence"], "stakes": adv["stakes"], "survive": adv["survive"],
            },
        }


def describe(sim: Sim, opt: Option | None) -> str:
    if opt is None:
        return ""
    ai, tid = opt
    if ai == POOL:
        return "wait"
    name = sim.R.abilities[ai].name
    e = sim.enemy(tid) if tid >= 0 else None
    return f"{name} → {e.name}" if e is not None and len(sim.enemies) > 1 else name


def dot_name(sim: Sim, key: int) -> str:
    R = sim.R
    return R.abilities[key].name if key < R.n_abilities else R.procs[key - R.n_abilities].id.replace("_", " ").title()


def context(sim: Sim) -> str:
    """A one-line summary of the state for the review list."""
    R, t = sim.R, sim.t
    tgt = sim.enemy(sim.target)
    parts = []
    for i, r in enumerate(R.res):
        v = sim.points(i, tgt) if i in R.on_target else sim.res[i]
        parts.append(f"{r.id.replace('_', ' ')} {v:.0f}")
    parts += [f"{b.name} {sim.buff_until[i] - t:.0f}s" for i, b in enumerate(R.buffs) if sim.buff_until[i] > t]
    if tgt is not None:
        parts.append(f"target {100 * tgt.hp / tgt.hp_max:.0f}%")
        parts += [f"{dot_name(sim, k)} {nt - t + (left - 1) * iv:.0f}s" for k, (nt, left, _, iv, _) in tgt.dots.items()]
        if not sim.behind(tgt):
            parts.append("not behind")
    return ", ".join(parts)


def why_not(sim: Sim, opt: Option) -> str:
    """Why an option isn't legal right now, in a few words."""
    ai, tid = opt
    R, t = sim.R, sim.t
    a = R.abilities[ai]
    if sim.knocked_until > t:
        return "knocked back"
    if sim.cd_until[ai] > t + EPS or (R.group[ai] >= 0 and sim.group_until[R.group[ai]] > t + EPS):
        return f"{a.name} is on cooldown"
    if R.req_buff[ai] >= 0 and not sim.buff_until[R.req_buff[ai]] > t:
        return f"needs {R.buffs[R.req_buff[ai]].name}"
    if R.req_no_buff[ai] >= 0 and sim.buff_until[R.req_no_buff[ai]] > t:
        return f"not during {R.buffs[R.req_no_buff[ai]].name}"
    if not sim.affordable(ai):
        short = [R.res_ids[r].replace("_", " ") for r, amt in sim.cost_of(ai) if sim.res[r] + EPS < amt]
        return f"not enough {', '.join(short)}"
    if a.next_swing and sim.queued >= 0:
        return "already queued"
    e = sim.enemy(tid) if tid >= 0 else None
    if R.targeted[ai]:
        if e is None:
            return "no target"
        if not sim.in_range(ai, e):
            return "out of range" if not a.gap_closer else "too close or too far"
        if a.req_behind and not sim.behind(e):
            return "must be behind the target"
        if a.req_target_health_below and e.hp >= a.req_target_health_below * e.hp_max:
            return f"target must be below {100 * a.req_target_health_below:.0f}%"
        fin = R.finisher[ai]
        if fin >= 0 and sim.points(fin, e) < 1:
            return f"no {R.res_ids[fin].replace('_', ' ')} on this target"
    return "can't use that now"
