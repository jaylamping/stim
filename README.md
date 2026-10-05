# stim

stim is a rotation helper for WoW Forever, trained in a combat simulator. In game it's an addon,
Stim, that shows your next few spells. Feral Druid is the first spec.

## How it works

- **Simulator** (`src/stim/sim.py`): melee combat with swings, resources, procs, auras, DoTs, and
  positioning. Nothing in it is class-specific; each spec is a TOML file in `specs/`.
- **Teacher** (`src/stim/teacher.py`): to label a decision, it simulates the next 20 seconds 16
  times for every usable ability, with different random rolls each time, and compares the average
  damage. It's too slow to run live.
- **Model** (`src/stim/model.py`): a small transformer trained to imitate the teacher, at about 1 ms
  per decision. It's then retrained on fights it played itself, with the teacher labeling its
  decisions.
- **Addon** (`addon/Stim`): a much smaller network, trained only on what an addon can see in
  combat on Forever.

## Specs

- **Feral Druid:** the WoW Forever beta kit after Blizzard's Oct 1 changes, with numbers datamined
  from the beta client where known.
- **Combat Rogue** and **Fury Warrior:** placeholder numbers.

## Stim

The addon shows your next three spells next to your character. It's Feral only for now.

During combat, Forever hides energy, combo points, buffs, cooldowns, and target health from addons.
Stim tracks what it can from your own casts, target changes, range checks, and the clock,
starting from values it reads before the pull. For energy, it computes a queue for each fifth of
the bar and lets the game show the one that matches, so the addon never reads your energy. It only
shows icons: it doesn't press anything or read the game any other way.

Install instructions and details are in [addon/Stim/README.md](addon/Stim/README.md).

## Quickstart

```bash
uv sync
uv run pytest
uv run python scripts/baselines.py
```

The commands for the full pipeline (data generation, training, evaluation, addon export) are in
[CLAUDE.md](CLAUDE.md).

## Status

These results are all from the simulator.

- For Rogue and Warrior, the model matches or beats the hand-written priority lists in every fight
  type we test (single boss, boss with adds, trash). Feral's kit changed after it was trained, so
  it hasn't been compared yet.
- Stim, using only what Forever lets addons see, is within 2% of a priority list that sees
  everything on a single boss, and a little ahead with adds and on trash. It hasn't been tried in
  the real client yet.
- A practice trainer that runs in the browser (`stim.trainer`) works but is shelved.

[HANDOFF.md](HANDOFF.md) has the full notes, numbers, and next steps.
