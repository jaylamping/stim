# stim

A simulation-trained rotation brain for World of Warcraft.

Hekili-style helpers are hand-written priority lists. stim works differently. A search-based
teacher simulates ahead to find the best ability in each situation, and a small Jev-style decision
model learns to make the same call in about a millisecond. The game rules are written down; the
strategy is learned.

The engine is class-agnostic: each class/spec is a TOML file in `specs/` that composes generic
mechanics (resources, swings, procs, auras, DoTs, positioning). It ships with Feral Druid (the WoW
Forever beta kit after Blizzard's Oct 1 changes), Combat Rogue, and Fury Warrior. Spell numbers
come from datamined beta client data where known; the rest are placeholders.

## In the game: StimCoach

[`addon/StimCoach`](addon/StimCoach) flashes your next three spells next to your character (Feral
for now). WoW Forever hides most combat values from addons ("secret values"), so StimCoach works
only from what the game allows, mainly your own casts and resource values from before the pull.
For energy, it precomputes a recommendation per energy band and lets the game show the right one.
It never reads the client any other way, and every button press is yours.

## Quickstart

```bash
uv sync
uv run pytest
uv run python scripts/baselines.py
```

The full pipeline (teacher data, training, evaluation, addon export) is in [CLAUDE.md](CLAUDE.md)
and [addon/StimCoach/README.md](addon/StimCoach/README.md).

## Status

Done:
- the engine, specs, and encounters
- the rollout teacher, the Jev-style model, and expert iteration
- the StimCoach addon (Feral)
- a practice trainer, shelved for now

Students beat the hand-written priority lists in most fights. StimCoach plays about even with a
full-information priority list in the sim while seeing only what Forever allows; it's untested in
the real client. See [HANDOFF.md](HANDOFF.md).
