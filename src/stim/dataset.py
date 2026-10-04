"""Training examples: teacher records turned into tokens and targets, cached per run."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features as F
from .data import load_episodes, restore
from .teacher import TeacherConfig, summarize


@dataclass
class Example:
    tok: F.Tokens
    soft: np.ndarray  # (options,) teacher soft target
    gap: np.ndarray  # (options,) teacher regret, in seconds of reference DPS
    best: int
    survive: float  # P(current target alive 12 s later), averaged over options by the soft target
    level: int  # stakes level
    baselines: dict[str, int]  # option index each baseline picks
    spec_name: str
    kind: str
    episode: int  # index of the episode across the runs it was built from


def examples_from_run(run_dir: str | Path, cfg: TeacherConfig | None = None, episode_offset: int = 0) -> list[Example]:
    """Featurize every record in a run, caching the result next to the shards."""
    cfg = cfg or TeacherConfig()
    run_dir = Path(run_dir)
    cache = run_dir / f"examples_v{F.VERSION}_tau{cfg.tau_floor:g}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            out = pickle.load(f)
        for ex in out:
            ex.episode += episode_offset
        return out
    out = []
    for ei, ep in enumerate(load_episodes(run_dir)):
        ref = ep.spec.reference_dps
        for r in ep.records:
            sim = restore(r.state, ep.spec)
            s = summarize(r.q.astype(np.float64), ref, cfg)
            soft = s["soft"].astype(np.float32)
            out.append(Example(
                tok=F.tokens(sim, r.options),
                soft=soft,
                gap=(s["gap"] / ref).astype(np.float32),
                best=s["best"],
                survive=min(1.0, float((soft[:, None] * r.alive).sum() / r.alive.shape[1])),
                level=s["stakes_level"],
                baselines=r.baselines,
                spec_name=ep.spec_name,
                kind=ep.kind,
                episode=ei,
            ))
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    for ex in out:
        ex.episode += episode_offset
    return out


def split_by_episode(examples: list[Example], val_frac: float = 0.1, seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Validation holds out whole episodes, so near-duplicate states from one fight never straddle the split."""
    episodes = sorted({ex.episode for ex in examples})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(episodes, size=max(1, int(len(episodes) * val_frac)), replace=False).tolist())
    train = [ex for ex in examples if ex.episode not in held]
    val = [ex for ex in examples if ex.episode in held]
    return train, val
