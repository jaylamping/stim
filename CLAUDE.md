# stim

A simulation-trained rotation brain for WoW, WoW Forever first. Read `HANDOFF.md` for the full
context, research findings, and the plan.

## Ground rules

- Keep the engine class-agnostic. No class- or spec-specific code paths in `src/stim`. Class
  behavior lives in `specs/*.toml`. If a spec needs a new mechanic, add it as a generic primitive
  in `spec.py` (schema and validation) and `sim.py`, with a test.
- Support melee classes first. Casters and healers come later.
- Never build anything that reads the live game client (screen capture, memory) or sends input to
  it. WoW Forever's addon API deliberately blocks in-combat decision-making, so stim runs outside
  the game as a practice trainer and post-pull combat-log coach.
- Priority lists (`[[apl]]` in specs) are baselines to beat, not the product.
- The teacher must never see future scripted events. Use `Sim.resample_future` in rollouts.

## Commands

```bash
uv sync
uv run pytest
uv run python scripts/baselines.py [episodes]
```
