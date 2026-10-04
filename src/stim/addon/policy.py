"""The in-game addon's policy: a small MLP from tracker features to a score for each ability and for
waiting, small enough to run in Lua every frame. Trained on soft teacher targets, and evaluated in the
sim by `AddonPolicy`, which plays only on what the addon could know."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..sim import POOL, Sim, rules_for
from ..teacher import TeacherConfig, summarize
from .tracker import action_of


class AddonNet(nn.Module):
    def __init__(self, width: int, actions: int, hidden: int = 64, mean=None, std=None):
        super().__init__()
        self.width, self.actions, self.hidden = width, actions, hidden
        self.register_buffer("mean", torch.zeros(width) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(width) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, actions))

    def forward(self, x: torch.Tensor, possible: torch.Tensor) -> torch.Tensor:
        logits = self.net((x - self.mean) / self.std)
        return logits.masked_fill(~possible, float("-inf"))


@dataclass
class AddonExample:
    obs: np.ndarray
    possible: np.ndarray
    soft: np.ndarray  # (actions,) teacher soft target over the addon's actions
    gap: np.ndarray  # (actions,) regret in seconds of reference DPS against the best current-target option; inf if illegal
    gap_all: float  # the best current-target option's regret against the best option on any target
    apl: int  # the priority list's action, or -1 if it picked another target
    spec_name: str
    kind: str
    episode: int


def gated_obs(r, spec) -> np.ndarray:
    """A record's features as the addon shows them in game: the gated resource set to the band its true
    value falls in (the game picks that band's recommendation), everything else from the tracker."""
    tr = r.tracker
    tr.spec, tr.R = spec, rules_for(spec)
    known = tr.band_of(r.state["res"][tr.gated]) if tr.gated >= 0 else None
    return tr.features(*r.ranges, known=known)


def addon_examples(episodes, cfg: TeacherConfig | None = None, gated: bool = True) -> list[AddonExample]:
    cfg = cfg or TeacherConfig()
    out = []
    for ei, ep in enumerate(episodes):
        A = len(ep.spec.abilities)
        ref = ep.spec.reference_dps
        for r in ep.records:
            if r.obs is None or (gated and r.tracker is None):
                continue
            target = r.state["target"]
            acts = [action_of(o, target, A) for o in r.options]
            keep = [i for i, a in enumerate(acts) if a >= 0]
            if len(keep) < 2:
                continue
            q = r.q.astype(np.float64)
            full = summarize(q, ref, cfg)
            sub = summarize(q[keep], ref, cfg)
            soft = np.zeros(A + 1, dtype=np.float32)
            gap = np.full(A + 1, np.inf, dtype=np.float32)
            for j, i in enumerate(keep):
                soft[acts[i]] = sub["soft"][j]
                gap[acts[i]] = sub["gap"][j] / ref
            possible = r.possible.copy()
            possible |= soft > 0  # the teacher knows better than the tracker's certainty mask
            apl_opt = r.options[r.baselines["apl"]]
            obs = gated_obs(r, ep.spec) if gated else r.obs
            out.append(AddonExample(obs, possible, soft, gap, float(full["gap"][keep[sub["best"]]] / ref),
                                    action_of(apl_opt, target, A), ep.spec_name, ep.kind, ei))
    return out


class AddonPolicy:
    """Plays on the addon's knowledge only: a tracker fed by the sim's events, and the net's scores. It
    behaves like a player following the addon: press the top suggestion when the game allows it; if
    you can't afford it yet, wait for the button to light up; if it's impossible for another reason
    (you're not behind the target, out of range), take the next suggestion."""

    name = "addon"

    def __init__(self, net: AddonNet, gated: bool = True):
        self.net = net.eval()
        self.gated = gated
        self.view = None

    def ranked(self, sim: Sim) -> list[int]:
        from ..data import AddonView

        if self.view is None or self.view.sim is not sim:
            self.view = AddonView(sim)
        obs, possible = self.view.observe()
        tr = self.view.tracker
        if self.gated and tr.gated >= 0:
            # the game shows the recommendation for the band your real value is in
            melee, charge = self.view.ranges()
            obs = tr.features(melee, charge, known=tr.band_of(sim.res[tr.gated]))
        with torch.inference_mode():
            logits = self.net(torch.from_numpy(obs)[None], torch.from_numpy(possible)[None])[0]
        return [int(i) for i in torch.argsort(logits, descending=True) if torch.isfinite(logits[i])]

    def __call__(self, sim: Sim, opts: list[tuple[int, int]]) -> tuple[int, int]:
        if len(opts) == 1:
            return opts[0]
        A = sim.R.n_abilities
        by_action = {action_of(o, sim.target, A): o for o in opts}
        for a in self.ranked(sim):
            if a in by_action:
                return by_action[a]
            if a < A and not sim.affordable(a) and sim.cd_until[a] <= sim.t:
                return (POOL, -1)  # it's coming: wait for the energy
        return (POOL, -1)


def save_addon(path, net: AddonNet, spec_name: str, layout: list[str], extra: dict | None = None) -> None:
    torch.save({"width": net.width, "actions": net.actions, "hidden": net.hidden, "state": net.state_dict(),
                "spec": spec_name, "layout": layout, **(extra or {})}, path)


def load_addon(path) -> tuple[AddonNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = AddonNet(ck["width"], ck["actions"], ck["hidden"])
    net.load_state_dict(ck["state"])
    return net.eval(), ck
