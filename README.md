# stim

A simulation-trained rotation brain for World of Warcraft.

Hekili-style helpers are hand-written priority lists. stim works differently. A search-based
teacher simulates ahead to find the best ability in each situation, and a small Jev-style decision
model learns to make the same call in about a millisecond. The game rules are written down; the
strategy is learned.

The engine is class-agnostic: each class/spec is a TOML file in `specs/` that composes generic
mechanics (resources, swings, procs, auras, DoTs, positioning). It ships with Feral Druid (the WoW
Forever beta kit), Combat Rogue, and Fury Warrior. All numbers are placeholders for now.

## Why it runs outside the game

WoW Forever ships Midnight's addon restrictions. In combat, health, auras, and even combo points
come back as secret values that addons can display but can't compute with. So stim is a practice
trainer (and later a post-pull coach built on combat logs). It never reads the game client or sends
input to it.

## Quickstart

```bash
uv sync
uv run pytest
uv run python scripts/baselines.py
```

## Status

The combat engine, specs, encounter generators, and baseline policies are done. The teacher, the
model, training, and the practice trainer are next. See [HANDOFF.md](HANDOFF.md).
