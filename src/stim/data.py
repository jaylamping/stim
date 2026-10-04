"""Labeled decision states: playing episodes with a behavior policy, labeling states with the teacher,
and storing them.

A record keeps a snapshot of the decision state rather than features, so features can change
without relabeling. Snapshots drop everything the player couldn't know (scripted future events and
combat rolls) and the spec, which is stored once per episode.
"""

from __future__ import annotations

import copy
import functools
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .policies import AplPolicy, EpsilonPolicy, GreedyPolicy, best_baseline
from .scenarios import make_scenario
from .sim import Sim, rules_for
from .spec import Spec, load_spec
from .teacher import Label, Teacher, TeacherConfig

Option = tuple[int, int]
_DROP = ("spec", "R", "streams", "events", "log", "cast_log")


@dataclass
class Record:
    """One labeled decision state."""

    state: dict  # `snapshot` of the sim
    options: list[Option]  # legal options, in `Sim.legal_options` order
    q: np.ndarray  # (options, samples) float32 teacher values
    alive: np.ndarray  # (options, samples) bool: current target alive survive_after later
    baselines: dict[str, int]  # option index each baseline policy would pick
    obs: np.ndarray | None = None  # what the in-game addon's tracker knew (`addon.tracker.Tracker.features`)
    possible: np.ndarray | None = None  # actions the tracker knew were possible (abilities, then wait)
    tracker: object | None = None  # the tracker itself (spec stripped), to recompute features later
    ranges: tuple[bool, bool] | None = None  # (in melee range, in gap-closer range)


@dataclass
class Episode:
    spec: Spec  # the (randomized) spec the episode was played with
    spec_name: str
    kind: str
    seed: int
    behavior: str
    records: list[Record] = field(default_factory=list)
    dps: float = 0.0


def snapshot(sim: Sim) -> dict:
    """The decision state as plain data, minus the spec and anything about the future."""
    state = {k: v for k, v in sim.__dict__.items() if k not in _DROP}
    state["enemies"] = [e.clone() for e in sim.enemies]
    for k in ("res", "res_target", "next_tick", "swing", "cd_until", "group_until", "buff_until",
              "buff_stacks", "buff_charges", "dmg_by"):
        state[k] = list(state[k])
    state["last_event"] = dict(sim.last_event)
    state["ev_i"] = 0
    return state


def restore(state: dict, spec: Spec, seed: int = 0) -> Sim:
    """A live sim from a snapshot. Its future holds no scripted events."""
    sim = Sim.__new__(Sim)
    sim.__dict__.update(copy.copy(state))
    sim.enemies = [e.clone() for e in state["enemies"]]
    for k in ("res", "res_target", "next_tick", "swing", "cd_until", "group_until", "buff_until",
              "buff_stacks", "buff_charges", "dmg_by"):
        setattr(sim, k, list(state[k]))
    sim.last_event = dict(state["last_event"])
    sim.spec, sim.R = spec, rules_for(spec)
    sim.events, sim.log, sim.cast_log = [], None, None
    sim.reseed(seed)
    return sim


def _portable(spec: Spec) -> Spec:
    s = copy.copy(spec)
    s.__dict__.pop("_rules", None)
    return s


@dataclass
class GenConfig:
    samples: int = 16
    horizon: float = 20.0
    label_prob: float = 0.3  # chance to label each decision with more than one legal option
    epsilon: float = 0.15  # random actions in the behavior policy, at unlabeled states
    randomize: float = 1.0  # strength of per-episode spec randomization (0 = off)
    rollout: str = "auto"  # "auto" (better of greedy and APL), "greedy", "apl", or "student=<checkpoint>"
    behavior: str = "greedy,apl"  # policies an episode picks its behavior from (same names as rollout)


@functools.cache
def default_rollout(spec_name: str) -> str:
    """The better baseline for a spec, chosen once on its unrandomized numbers."""
    return best_baseline(load_spec(spec_name)).name


_students: dict[str, object] = {}


def named_policy(spec: Spec, name: str):
    """greedy, apl, or student=<checkpoint> (loaded once per process, on one CPU thread)."""
    if name.startswith("student="):
        path = name.split("=", 1)[1]
        if path not in _students:
            import torch

            from .student import load_student

            torch.set_num_threads(1)
            _students[path] = load_student(path)
        return _students[path]
    return AplPolicy(spec) if name == "apl" else GreedyPolicy()


def generate_episode(spec_name: str, kind: str, seed: int, cfg: GenConfig, rollout: str | None = None) -> Episode:
    """Play one episode, labeling a share of its decision states with the teacher. At a labeled state
    the episode follows the teacher's best option; elsewhere it follows a behavior policy (one of
    `cfg.behavior`, chosen per episode) with epsilon random actions."""
    rng = random.Random(seed)
    base = load_spec(spec_name)
    spec = base.randomized(rng, cfg.randomize) if cfg.randomize > 0 else base
    rollout = rollout or (cfg.rollout if cfg.rollout != "auto" else default_rollout(spec_name))
    teacher = Teacher(named_policy(spec, rollout), TeacherConfig(samples=cfg.samples, horizon=cfg.horizon))
    baselines = {"greedy": GreedyPolicy(), "apl": AplPolicy(spec)}
    choice = rng.choice(cfg.behavior.split(","))
    behavior = EpsilonPolicy(named_policy(spec, choice), cfg.epsilon, random.Random(rng.getrandbits(32)))
    sim = make_scenario(spec, kind, random.Random(rng.getrandbits(32)))
    ep = Episode(_portable(spec), spec_name, kind, seed, behavior.name)
    watch = AddonView(sim)
    while not sim.done:
        opts = sim.legal_options()
        if len(opts) > 1 and rng.random() < cfg.label_prob:
            label: Label = teacher.evaluate(sim, rng, opts)
            obs, possible = watch.observe()
            tracker = watch.tracker.copy()
            tracker.spec = tracker.R = None
            ep.records.append(Record(
                state=snapshot(sim),
                options=list(opts),
                q=label.q.astype(np.float32),
                alive=label.alive,
                baselines={k: opts.index(p(sim, opts)) for k, p in baselines.items()},
                obs=obs,
                possible=possible,
                tracker=tracker,
                ranges=watch.ranges(),
            ))
            sim.step(opts[label.best])
        else:
            sim.step(behavior(sim, opts))
        watch.update()
    ep.dps = sim.damage / sim.t
    return ep


class AddonView:
    """Feeds a sim's events to an addon tracker the way the game feeds the addon: your casts, target
    changes and deaths, and range checks. Nothing else about the sim reaches the tracker."""

    def __init__(self, sim: Sim):
        from .addon.tracker import Tracker

        self.sim = sim
        sim.cast_log = []
        self.seen = 0
        self.target = sim.target
        self.tracker = Tracker(sim.spec)
        self.tracker.start(sim.t, list(sim.res))

    def update(self) -> None:
        sim, tr = self.sim, self.tracker
        for t, ai, tid in sim.cast_log[self.seen:]:
            if tid >= 0 and tid != self.target:
                tr.retarget(t)
                self.target = tid
            tr.cast(t, ai)
        self.seen = len(sim.cast_log)
        if sim.target != self.target:
            tr.retarget(sim.t)
            self.target = sim.target
        tr.advance(sim.t)

    def ranges(self) -> tuple[bool, bool]:
        sim = self.sim
        e = sim.enemy(sim.target)
        if e is None:
            return False, False
        d = sim.dist(e)
        charge = any(a.gap_closer and a.range_min <= d <= a.range_max for a in sim.R.abilities)
        return d <= sim.spec.melee_range, charge

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        self.update()
        melee, charge = self.ranges()
        return self.tracker.features(melee, charge), self.tracker.possible(melee, charge)


def save_shard(path: str | Path, episodes: list[Episode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_episodes(run_dir: str | Path) -> list[Episode]:
    """Every episode in a run directory (shards are local, trusted pickle files)."""
    episodes: list[Episode] = []
    for p in sorted(Path(run_dir).glob("shard_*.pkl")):
        with open(p, "rb") as f:
            episodes.extend(pickle.load(f))
    return episodes

