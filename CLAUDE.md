# stim

A simulation-trained rotation brain for WoW, WoW Forever first. Read `HANDOFF.md` for the full
context, research findings, and the plan.

## Ground rules

- Keep the engine class-agnostic. No class- or spec-specific code paths in `src/stim`. Class
  behavior lives in `specs/*.toml`. If a spec needs a new mechanic, add it as a generic primitive
  in `spec.py` (schema and validation) and `sim.py`, with a test.
- Support melee classes first. Casters and healers come later.
- In the game, stim is an addon that uses only the official addon API, and every button press is
  the player's own. Never automate input, and never read the client any other way (screen capture,
  memory). Forever hides most combat values from addons ("secret values"): work with what the API
  gives in combat (your own casts, target death, range checks, the swing timer, the clock) plus what
  it gives out of combat, and reconstruct the rest from the spec's rules. Don't pry hidden values
  out of the game's own widgets.
- Outside the game: the practice trainer (`stim.trainer`, shelved for now) and a post-pull
  combat-log coach.
- Priority lists (`[[apl]]` in specs) are baselines to beat, not the product.
- The teacher must never see the future: neither scripted events nor combat rolls. Rollouts get a
  fresh event draw (`Teacher.future`, or `Sim.resample_future`) and fresh rolls (`Sim.clone(seed)`).
- Labels are soft. Never train on hard labels; store raw rollout values so targets can be recomputed.

## Commands

```bash
uv sync
uv run pytest
uv run python scripts/baselines.py [episodes]
uv run python scripts/calibrate_values.py --policy apl          # measured resource exchange rates
uv run python scripts/gen_data.py --out runs/gen0 --episodes 350  # teacher-labeled states
uv run python scripts/train.py --data runs/gen0 --out runs/gen0/student.pt
uv run python scripts/evaluate.py --policies greedy apl student=runs/gen0/student.pt
# the in-game addon (addon/Stim): data with tracker views, a small policy, then Data.lua
uv run python scripts/gen_data.py --out runs/addon1 --specs feral --episodes 700
uv run python scripts/train_addon.py --data runs/addon1 --spec feral --out runs/addon1/addon_feral.pt
uv run python scripts/export_addon.py --spec feral --model runs/addon1/addon_feral.pt
```

The Lua addon mirrors `src/stim/addon/tracker.py` and the policy exactly; `tests/test_addon.py`
checks parity, so change both sides together.

`runs/` is gitignored: datasets, caches, and checkpoints live there.
